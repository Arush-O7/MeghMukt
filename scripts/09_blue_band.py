import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.bluesynth import BlueSynth, ratio_features
from meghmukt.io import load_config, write_raster

ITEM_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{}"


def download_blue(cfg):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "01_download_sentinel2.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    for aoi, a in cfg["sentinel2"]["aois"].items():
        d = cfg["paths"]["raw"] / "sentinel2" / aoi
        meta = json.loads((d / "meta.json").read_text())
        for f in meta["files"]:
            if f["kind"] != "clear":
                continue
            out = d / f["file"].replace(".tif", "_blue.tif")
            if out.exists():
                continue
            it = requests.get(ITEM_URL.format(f["item"]), timeout=60).json()
            with rasterio.open(d / f["file"]) as src:
                bounds, prof = src.bounds, src.profile
            arr, _ = dl.read_window(it["assets"]["blue"]["href"], tuple(bounds), prof["width"])
            write_raster(out, arr[None].astype(np.uint16), dict(driver="GTiff", crs=prof["crs"], transform=prof["transform"],
                                                                width=prof["width"], height=prof["height"]),
                         tags={"source": it["id"], "scale_offset": json.dumps(dl.scale_offset(it, "blue"))}, band_names=["blue_B02"])
            print(f"  blue {aoi} {f['date']}")


def load_pixels(cfg, aois, n_per_chip, rng):
    X, B = [], []
    for aoi in aois:
        d = cfg["paths"]["raw"] / "sentinel2" / aoi
        for f in json.loads((d / "meta.json").read_text())["files"]:
            if f["kind"] != "clear":
                continue
            with rasterio.open(d / f["file"]) as s:
                grn = s.read([1, 2, 3]).astype(np.float32)
                so = json.loads(s.tags()["scale_offset"])
            with rasterio.open(d / f["file"].replace(".tif", "_blue.tif")) as s:
                b = s.read(1).astype(np.float32)
                bso = json.loads(s.tags()["scale_offset"])
            refl = np.stack([grn[i] * so[i][0] + so[i][1] for i in range(3)]).reshape(3, -1)
            blue = (b * bso[0] + bso[1]).ravel()
            ok = (refl > 0.001).all(0) & (blue > 0.001)
            idx = rng.choice(np.flatnonzero(ok), min(n_per_chip, ok.sum()), replace=False)
            X.append(refl[:, idx]), B.append(blue[idx])
    return np.concatenate(X, 1), np.concatenate(B)


def baselines(X):
    g, r, n = X
    ndvi, ndwi = np.clip((n - r) / (n + r + 1e-6), 0, 1), (g - n) / (g + n + 1e-6)
    out = {"current": np.clip(0.85 * g + 0.25 * r - 0.12 * n, 0, None),
           "ndvi_ndwi_ratio": g * np.clip(0.912 + 0.320 * ndvi + 0.935 * ndwi, 0.35, 1.30)}
    w = np.array([0.62, 0.35, -0.08, 0.12, 0.05, -0.02, -0.09, 0.03, 0.01, 0.005])
    feats = np.stack([g, r, n, g * g, r * r, n * n, g * r, r * n, g * n, np.ones_like(g)])
    out["fixed_poly"] = np.clip(w @ feats, 0, None)
    return out


def scores(pred, true):
    err = pred - true
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)),
            "ratio_mae": float(np.mean(np.abs(pred / (true + 1e-6) - 1)))}


def main():
    cfg = load_config()
    P = cfg["paths"]
    rng = np.random.default_rng(0)
    print("[blue] downloading Sentinel-2 B02 for the clear chips (first run only) ...")
    download_blue(cfg)
    Xtr, Btr = load_pixels(cfg, cfg["split"]["train"], 20000, rng)
    Xte, Bte = load_pixels(cfg, cfg["split"]["test"] + cfg["split"]["val"], 20000, rng)
    print(f"[blue] fit on {Xtr.shape[1]} px (train areas), test on {Xte.shape[1]} px (unseen areas)")

    results = {k: scores(v, Bte) for k, v in baselines(Xte).items()}
    fits = {}
    for kind in ("ratio", "mlp"):
        m = BlueSynth.fit(Xtr, Btr, kind=kind)
        results[f"{kind}-fit"] = scores(m.predict(Xte), Bte)
        fits[kind] = m
    for k, v in results.items():
        print(f"  {k:10s} RMSE {v['rmse']:.4f}  R² {v['r2']:+.3f}  mean |B/B_true - 1| {v['ratio_mae']:.1%}")
    best = min(("ratio", "mlp"), key=lambda k: results[f"{k}-fit"]["rmse"])
    if best == "mlp" and results["mlp-fit"]["rmse"] > 0.9 * results["ratio-fit"]["rmse"]:
        best = "ratio"
    fits[best].save(P["processed"] / "blue_fit.json")
    (P["results"] / "blue_band_study.json").write_text(json.dumps(
        {"chosen": f"{best}-fit", "train_areas": cfg["split"]["train"],
         "test_areas": cfg["split"]["test"] + cfg["split"]["val"], "results": results}, indent=2))
    print(f"[blue] chosen: {best}-fit -> data/processed/blue_fit.json")

    aoi = cfg["split"]["test"][0]
    d = P["raw"] / "sentinel2" / aoi
    f = [x for x in json.loads((d / "meta.json").read_text())["files"] if x["kind"] == "clear"][0]
    with rasterio.open(d / f["file"]) as s:
        grn = s.read([1, 2, 3]).astype(np.float32)[:, 300:812, 300:812] * 1e-4
    with rasterio.open(d / f["file"].replace(".tif", "_blue.tif")) as s:
        blue = s.read(1).astype(np.float32)[300:812, 300:812] * 1e-4
    flat = grn.reshape(3, -1)
    cands = {"measured blue (truth)": blue, **{k: v.reshape(blue.shape) for k, v in baselines(flat).items()},
             f"{best}-fit (MeghMukt)": fits[best].predict(flat).reshape(blue.shape)}
    lo, hi = np.percentile(np.stack([grn[1], grn[0], blue]), [2, 98])
    fig, ax = plt.subplots(1, len(cands), figsize=(3.4 * len(cands), 3.8))
    for a, (t, b) in zip(ax, cands.items()):
        rgb = np.clip((np.stack([grn[1], grn[0], b], -1) - lo) / (hi - lo), 0, 1) ** 0.85
        a.imshow(rgb), a.set_title(t, fontsize=9), a.axis("off")
    fig.suptitle(f"Natural colour of an unseen area ({aoi}) — only the blue channel differs", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    (P["results"] / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(P["results"] / "figures" / "blue_band.png", dpi=120)


if __name__ == "__main__":
    main()
