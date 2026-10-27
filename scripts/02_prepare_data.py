import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import Normaliser, SyntheticCloudDataset, fake_coarse_reference, make_condition
from meghmukt.io import Harmoniser, liss4_dn_to_norm, load_config, read_raster
from meghmukt.masking import CLOUD, SHADOW, MaskParams, detect, scl_to_mask
from meghmukt.synth import CloudSynthesiser

LISS4_TRAIN = ["model_1_20240414_20240426_GTDB_0003_0046.npy", "model_2_20240419_20240426_GTDA_2_0019_0014.npy"]
LISS4_TEST = ["model_3_20240414_20240426_GTDB_0001_0068.npy"]
LISS4_REALCLOUD = ["model_4_20240419_20240426_GTDA_2_0006_0018.npy", "R2F13JUN2024068249010700_cloudy.tif"]


def load_s2_chip(path):
    with rasterio.open(path) as src:
        dn = src.read().astype(np.float32)
        so = json.loads(src.tags()["scale_offset"])
        profile = src.profile
    refl = np.stack([dn[i] * so[i][0] + so[i][1] for i in range(3)])
    return np.clip(refl, 0, 1.5), dn[3].astype(np.uint8), profile


def main():
    cfg = load_config()
    P = cfg["paths"]
    out = P["processed"]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["train"]["seed"])

    s2 = {}
    for aoi in cfg["sentinel2"]["aois"]:
        d = P["raw"] / "sentinel2" / aoi
        meta = json.loads((d / "meta.json").read_text())
        clear, cloudy, scl, dates_clear, dates_cloudy = [], [], [], [], []
        for f in meta["files"]:
            refl, s, _ = load_s2_chip(d / f["file"])
            if f["kind"] == "clear":
                clear.append(refl), dates_clear.append(f["date"])
            else:
                cloudy.append(refl), scl.append(s), dates_cloudy.append(f["date"])
        s2[aoi] = dict(clear=np.stack(clear), cloudy=np.stack(cloudy) if cloudy else None,
                       scl=np.stack(scl) if scl else None, dates_clear=dates_clear, dates_cloudy=dates_cloudy)
        print(f"[S2] {aoi:10s} clear={len(clear)} cloudy={len(cloudy)}")

    lroot = P["liss4_samples"]
    liss_train = np.stack([np.load(lroot / f) for f in LISS4_TRAIN]).astype(np.float32)
    liss_test = np.stack([np.load(lroot / f) for f in LISS4_TEST]).astype(np.float32)
    liss_cloudy_patch = np.load(lroot / LISS4_REALCLOUD[0]).astype(np.float32)
    tile_dn, tile_profile = read_raster(lroot / LISS4_REALCLOUD[1])
    liss_tile = liss4_dn_to_norm(tile_dn)
    print(f"[LISS-IV] train patches {liss_train.shape}, test {liss_test.shape}, cloudy tile {liss_tile.shape}")

    s2_ground = np.concatenate([s2[a]["clear"].transpose(1, 0, 2, 3).reshape(3, -1) for a in cfg["split"]["train"]], 1)
    s2_cloud = np.concatenate([x[:, s == 9] for a in cfg["split"]["train"] if s2[a]["cloudy"] is not None
                               for x, s in zip(s2[a]["cloudy"], s2[a]["scl"])], 1)
    l_ground = liss_train.transpose(1, 0, 2, 3).reshape(3, -1)
    tb = liss_tile.mean(0)
    l_cloud = liss_tile[:, tb >= np.quantile(tb, 0.5)]
    harm = Harmoniser.fit_anchors(s2_ground, s2_cloud, l_ground, l_cloud)
    harm.save(out / "harmoniser.json")
    print(f"[harmonise] S2 ground {np.median(s2_ground, 1).round(3)} cloud {np.median(s2_cloud, 1).round(3)} | "
          f"LISS-IV ground {np.median(l_ground, 1).round(3)} cloud {np.median(l_cloud, 1).round(3)}")
    print("[harmonise] gain", harm.gain.ravel().round(3), "bias", harm.bias.ravel().round(3))
    for a in s2:
        s2[a]["clear"] = np.stack([harm(x) for x in s2[a]["clear"]])
        if s2[a]["cloudy"] is not None:
            s2[a]["cloudy"] = np.stack([harm(x) for x in s2[a]["cloudy"]])
        np.savez_compressed(out / f"s2_{a}.npz", clear=s2[a]["clear"].astype(np.float16),
                            cloudy=(s2[a]["cloudy"] if s2[a]["cloudy"] is not None else np.zeros((0, 3, 1, 1))).astype(np.float16),
                            scl=s2[a]["scl"] if s2[a]["scl"] is not None else np.zeros((0, 1, 1), np.uint8),
                            dates_clear=np.array(s2[a]["dates_clear"]), dates_cloudy=np.array(s2[a]["dates_cloudy"]))
    np.savez_compressed(out / "liss4.npz", train=liss_train, test=liss_test, cloudy_patch=liss_cloudy_patch,
                        tile=liss_tile.astype(np.float32))

    def mask_scores(params, aois, classes=(CLOUD,)):
        tp = fp = fn = 0
        false_alarm = []
        for a in aois:
            if s2[a]["cloudy"] is not None:
                for x, s in zip(s2[a]["cloudy"], s2[a]["scl"]):
                    ref = np.isin(scl_to_mask(s), classes)
                    det = np.isin(detect(x, params)[0], classes)
                    tp += (ref & det).sum(); fp += (~ref & det).sum(); fn += (ref & ~det).sum()
            false_alarm += [np.isin(detect(x, params)[0], classes).mean() for x in s2[a]["clear"]]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        fa = float(np.mean(false_alarm))
        return dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                    f1=f1, false_alarm_clear=fa, score=f1 - 2 * fa)

    tr = cfg["split"]["train"]
    best = None
    for b in np.arange(0.18, 0.40, 0.02):
        for fl in (0.35, 0.5, 0.7):
            p = MaskParams(bright=round(float(b), 2), flatness=fl)
            sc = mask_scores(p, tr)
            if best is None or sc["score"] > best[1]["score"]:
                best = (p, sc)
    cloud_scores = best[1]
    for ratio in (0.6, 0.7, 0.8, 0.9):
        for reach in (60, 150, 300):
            for buf in (3, 6, 9):
                p = MaskParams(bright=best[0].bright, flatness=best[0].flatness, shadow_ratio=ratio,
                               shadow_reach=reach, buffer=buf)
                sc = mask_scores(p, tr, (CLOUD, SHADOW))
                if best[1].get("stage") != 2 or sc["score"] > best[1]["score"]:
                    best = (p, {**sc, "stage": 2})
    params = best[0]
    held = cfg["split"]["test"] + cfg["split"]["val"]
    test_cloud = mask_scores(params, held)
    test_shadow = mask_scores(params, held, (SHADOW,))
    test_all = mask_scores(params, held, (CLOUD, SHADOW))
    print(f"[mask] {params}")
    for n, t in (("cloud", test_cloud), ("shadow", test_shadow), ("cloud+shadow", test_all)):
        print(f"[mask] unseen areas, {n:12s}: F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} "
              f"false-alarm on clear dates={t['false_alarm_clear']:.2%}")
    f = lambda d: {k: float(v) for k, v in d.items()}
    mask_report = {"params": params.to_dict(), "reference": "Sentinel-2 SCL: 8/9/10 = cloud, 3 = shadow",
                   "train_cloud": f(cloud_scores), "train_cloud_shadow": f(best[1]),
                   "test_cloud": f(test_cloud), "test_shadow": f(test_shadow), "test_cloud_shadow": f(test_all)}
    (out / "mask_params.json").write_text(json.dumps(mask_report, indent=2))
    if "--mask-only" in sys.argv:
        return

    spectra = []
    for a in cfg["split"]["train"]:
        if s2[a]["cloudy"] is None:
            continue
        for x, s in zip(s2[a]["cloudy"], s2[a]["scl"]):
            px = x[:, s == 9].T
            if len(px):
                spectra.append(px[rng.choice(len(px), min(len(px), 300), replace=False)])
    tile_mask, _ = detect(liss_tile, params)
    px = liss_tile[:, tile_mask == CLOUD].T
    if len(px):
        spectra.append(px[rng.choice(len(px), min(len(px), 1500), replace=False)])
    spectra = np.concatenate(spectra).astype(np.float32)
    spectra = spectra[spectra.mean(1) > params.bright]
    np.save(out / "cloud_spectra.npy", spectra)
    print(f"[clouds] {len(spectra)} real cloud spectra, mean {spectra.mean(0).round(3)}")

    tr = np.concatenate([s2[a]["clear"].transpose(1, 0, 2, 3).reshape(3, -1) for a in cfg["split"]["train"]], 1)
    norm = Normaliser(tr.mean(1), tr.std(1))
    (out / "norm.json").write_text(json.dumps(norm.to_dict(), indent=2))
    print("[norm]", norm.to_dict())

    synth = CloudSynthesiser(spectra, tuple(cfg["data"]["cloud_cover_range"]), cfg["data"]["shadow_prob"])
    patch = cfg["data"]["patch_size"]
    for split, n_per in (("val", 96), ("test", 160)):
        sources = [{"clear": s2[a]["clear"]} for a in cfg["split"][split]]
        ds = SyntheticCloudDataset(sources, norm, synth, patch=patch, ref_dropout=0.0, seed=1000 + len(split), jitter=False)
        r = np.random.default_rng(7 if split == "val" else 11)
        items = [ds.sample(r) for _ in range(n_per * len(sources))]
        np.savez_compressed(out / f"eval_{split}.npz", cond=np.stack([c for c, _ in items]).astype(np.float16),
                            gt=np.stack([g for _, g in items]).astype(np.float16),
                            meta=json.dumps({"aois": cfg["split"][split], "patch": patch}))
        print(f"[eval] {split}: {len(items)} samples from {cfg['split'][split]}")

    r = np.random.default_rng(21)
    conds, gts = [], []
    for x in liss_test:
        for _ in range(12):
            cloudy, mask, _ = synth(x, r)
            conds.append(make_condition(cloudy, mask, fake_coarse_reference(x, r), norm, True))
            gts.append(norm(x))
    np.savez_compressed(out / "eval_liss4.npz", cond=np.stack(conds).astype(np.float16), gt=np.stack(gts).astype(np.float16),
                        meta=json.dumps({"source": LISS4_TEST, "note": "real LISS-IV ground truth, synthetic clouds"}))
    print(f"[eval] liss4: {len(conds)} samples (real LISS-IV patch, never used for training)")


if __name__ == "__main__":
    main()
