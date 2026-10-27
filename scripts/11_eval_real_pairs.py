import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import Normaliser, match_reference
from meghmukt.inference import reconstruct_scene
from meghmukt.io import Harmoniser, display_limits, get_device, load_config, to_display
from meghmukt.masking import MaskParams, detect_auto
from meghmukt.metrics import evaluate, fit_landcover, uncertainty_quality
from meghmukt.models import MODEL_NAMES, ClassicalFill, load_checkpoint
from meghmukt.perception import NIQE, patch_features, sharpness_ratio

MIN_CORR = 0.80
VAL_AREAS = {"silchar"}
LABELS = {"copy_input": "Do nothing", **MODEL_NAMES, "meghgan_noadv": "M2 without GAN loss",
          "meghgan_charb": "M2 + Charbonnier", "meghgan_real": "M2 MeghGAN + real-cloud training",
          "diffusion_real": "M3 Diffusion + real-cloud training"}


def load(path, harm):
    with rasterio.open(path) as s:
        dn = s.read().astype(np.float32)
        so = json.loads(s.tags()["scale_offset"])
    refl = np.stack([dn[i] * so[i][0] + so[i][1] for i in range(3)])
    return harm(np.clip(refl, 0, 1.5)), dn[3].astype(np.uint8)


def match(src, dst, ok):
    out = src.copy()
    for b in range(3):
        x, y = src[b][ok], dst[b][ok]
        a = np.cov(x, y, bias=True)[0, 1] / max(x.var(), 1e-9)
        out[b] = a * src[b] + (y.mean() - a * x.mean())
    return out


def main():
    cfg = load_config()
    P = cfg["paths"]
    dev = get_device(cfg["train"]["device"])
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    norm = Normaliser.load(P["processed"] / "norm.json")
    mp = MaskParams(**json.loads((P["processed"] / "mask_params.json").read_text())["params"])
    models = {"copy_input": None, "classical": ClassicalFill()}
    for t in ("unet", "meghgan", "meghgan_noadv", "meghgan_charb", "meghgan_real", "diffusion", "diffusion_real"):
        if (P["checkpoints"] / f"{t}.pt").exists():
            models[t] = load_checkpoint(P["checkpoints"] / f"{t}.pt", dev)[0]
    conf_model = models.get("diffusion_real", models.get("diffusion"))

    rows, pairs_info, shows = [], [], []
    niqe, feats = NIQE.load(P["processed"] / "niqe_pristine.npz"), {}
    for d in sorted((P["raw"] / "real_pairs").glob("*/meta.json")):
        meta = json.loads(d.read_text())
        if not meta.get("pairs") or not meta.get("reference") or meta["aoi"] in VAL_AREAS:
            continue
        ref, _ = load(d.parent / meta["reference"], harm)
        for pr in meta["pairs"]:
            cloudy, _ = load(d.parent / pr["cloudy"], harm)
            truth, scl_t = load(d.parent / pr["truth"], harm)
            truth_ok = ~np.isin(scl_t, (0, 1, 3, 8, 9, 10))
            name = f"{meta['aoi']} {pr['cloudy'][:10]}"
            for method in ("deep", "rules"):
                mask = detect_auto(cloudy, method, 10.0, mp)[0]
                both_clear = (mask == 0) & truth_ok
                truth_m = match(truth, cloudy, both_clear)
                corr = float(np.mean([np.corrcoef(truth_m[b][both_clear], cloudy[b][both_clear])[0, 1] for b in range(3)]))
                region = (mask > 0) & truth_ok
                ref_age = abs((date.fromisoformat(pr["cloudy"][:10]) - date.fromisoformat(meta["reference"][:10])).days)
                info = {"pair": name, "mask": method, "gap_days": pr["gap_days"], "reference_age_days": ref_age,
                        "ground_corr": round(corr, 3),
                        "region_frac": round(float(region.mean()), 3), "used": corr >= MIN_CORR and region.sum() > 500}
                pairs_info.append(info)
                if not info["used"]:
                    continue
                km = fit_landcover(truth_m)
                tags = list(models) if method == "deep" else [t for t in ("copy_input", "classical", "meghgan", "meghgan_real") if t in models]
                for t in tags:
                    if models[t] is None:
                        out, unc = cloudy, None
                    else:
                        out, unc, _ = reconstruct_scene(models[t], cloudy, mask, ref, norm, dev, tta=True,
                                                        conf_model=conf_model if t in ("meghgan", "meghgan_real") else None)
                    r = {"pair": name, "mask": method, "model": t, "reference_age_days": ref_age,
                         **evaluate(out, truth_m, region, km),
                         "sharpness_ratio": sharpness_ratio(out, truth_m, region)}
                    feats.setdefault((method, t), []).append(patch_features(out, region))
                    if t == "copy_input":
                        feats.setdefault((method, "_truth"), []).append(patch_features(truth_m, region))
                    if t.startswith("meghgan") and unc is not None:
                        r.update(uncertainty_quality(unc, np.abs(out - truth_m).mean(0), region))
                    rows.append(r)
                    if method == "deep" and t == ("meghgan_real" if "meghgan_real" in models else "meghgan"):
                        shows.append((name, cloudy, mask, match_reference(cloudy, mask, ref), out, truth_m, region))
                print(f"{name:22s} mask={method:5s} corr={corr:.2f} gap={pr['gap_days']}d  "
                      + "  ".join(f"{t}:{np.mean([x['psnr_cloud'] for x in rows if x['pair'] == name and x['mask'] == method and x['model'] == t]):.1f}"
                                  for t in tags), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(P["results"] / "real_pairs_per_pair.csv", index=False)
    pd.DataFrame(pairs_info).to_csv(P["results"] / "real_pairs_info.csv", index=False)
    cols = ["psnr_cloud", "ssim_cloud", "mae_cloud", "sam_cloud", "ndvi_mae_cloud", "lulc_agree_cloud", "sharpness_ratio",
            "unc_spearman"]
    summ = df.groupby(["mask", "model"], sort=False)[[c for c in cols if c in df]].mean().reset_index()
    summ["niqe_cloud"] = [niqe.score(np.concatenate(feats[(r.mask, r.model)])) for r in summ.itertuples()]
    niqe_truth = {m: niqe.score(np.concatenate(v)) for (m, t), v in feats.items() if t == "_truth"}
    summ["model_name"] = summ["model"].map(LABELS)
    summ.to_csv(P["results"] / "real_pairs_summary.csv", index=False)
    n_used = sum(i["used"] for i in pairs_info if i["mask"] == "deep")
    with open(P["results"] / "real_pairs_summary.md", "w") as f:
        f.write(f"Real clouds, {n_used} near-date pairs (gap <= 10 days, ground correlation >= {MIN_CORR}), unseen areas.\n")
        f.write(f"NIQE of the near-date truth on the same pixels: {niqe_truth}\n")
        for m, g in summ.groupby("mask", sort=False):
            f.write(f"\n### mask: {m}\n\n| Model | PSNR cloud | SSIM cloud | MAE | SAM ° | NDVI MAE | Land-cover | Sharpness | NIQE ↓ |\n|---|---|---|---|---|---|---|---|---|\n")
            for _, r in g.iterrows():
                f.write(f"| {r.model_name} | {r.psnr_cloud:.2f} | {r.ssim_cloud:.3f} | {r.mae_cloud:.4f} | {r.sam_cloud:.2f} | "
                        f"{r.ndvi_mae_cloud:.3f} | {100 * r.lulc_agree_cloud:.1f}% | {r.sharpness_ratio:.2f} | {r.niqe_cloud:.2f} |\n")
    print((P["results"] / "real_pairs_summary.md").read_text())

    shows = shows[:5]
    if shows:
        cols_ = ["Real cloudy input", "Deep mask", "Temporal reference (colour-matched)", "MeghGAN output",
                 "Clear pass ≤10 days later"]
        fig, ax = plt.subplots(len(shows), 5, figsize=(16, 3.3 * len(shows)), squeeze=False)
        for r_, (name, x, m, ref, out, tr, _) in enumerate(shows):
            lim = display_limits(tr)
            ims = [to_display(x), m, to_display(ref, limits=lim), to_display(out, limits=lim), to_display(tr, limits=lim)]
            for c, im in enumerate(ims):
                ax[r_, c].imshow(im, **({"cmap": "viridis", "vmin": 0, "vmax": 2, "interpolation": "nearest"} if c == 1 else {}))
                ax[r_, c].set_xticks([]), ax[r_, c].set_yticks([])
                if r_ == 0:
                    ax[r_, c].set_title(cols_[c], fontsize=10)
            ax[r_, 0].set_ylabel(name, fontsize=9)
        plt.tight_layout()
        plt.savefig(P["results"] / "figures" / "real_pairs.png", dpi=100)


if __name__ == "__main__":
    main()
