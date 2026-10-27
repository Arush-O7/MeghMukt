import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import Normaliser, match_reference
from meghmukt.inference import reconstruct_scene
from meghmukt.io import display_limits, get_device, load_config, to_display
from meghmukt.metrics import evaluate, fit_landcover
from meghmukt.models import MODEL_NAMES, ClassicalFill, load_checkpoint
from meghmukt.perception import NIQE, patch_features, sharpness_ratio
from meghmukt.synth import CloudSynthesiser

N_SYNTH = 48
LABELS = {"copy_input": "Do nothing", **MODEL_NAMES, "meghgan_liss4ft": "M2 + LISS-IV fine-tune (2 demo patches)",
          "meghgan_liss4": "M2 + real LISS-IV (Bhoonidhi) training"}


def main():
    cfg = load_config()
    P = cfg["paths"]
    dev = get_device(cfg["train"]["device"])
    norm = Normaliser.load(P["processed"] / "norm.json")
    niqe = NIQE.load(P["processed"] / "niqe_pristine.npz")
    models = {"copy_input": None, "classical": ClassicalFill()}
    for t in ("unet", "meghgan", "meghgan_real", "meghgan_liss4ft", "meghgan_liss4", "diffusion_real"):
        if (P["checkpoints"] / f"{t}.pt").exists():
            models[t] = load_checkpoint(P["checkpoints"] / f"{t}.pt", dev)[0]
    print("models:", list(models))

    cases = []
    synth = CloudSynthesiser(np.load(P["processed"] / "cloud_spectra.npy"), tuple(cfg["data"]["cloud_cover_range"]),
                             cfg["data"]["shadow_prob"])
    tc = np.load(P["processed"] / "liss4_test_clear.npz")
    rng = np.random.default_rng(2024)
    for key in list(tc.files)[:N_SYNTH]:
        stack = tc[key].astype(np.float32)
        t, r = rng.choice(len(stack), 2, replace=False)
        cloudy, mask, _ = synth(stack[t], rng)
        cases.append(dict(kind="A synthetic", cloudy=cloudy, mask=mask, truth=stack[t], ref=stack[r],
                          valid=np.ones(mask.shape, bool)))
    tp = np.load(P["processed"] / "liss4_test_pairs.npz")
    for i in range(len(tp["cloudy"])):
        cases.append(dict(kind="B real cloud", cloudy=tp["cloudy"][i].astype(np.float32), mask=tp["mask"][i],
                          truth=tp["truth"][i].astype(np.float32), ref=tp["ref"][i].astype(np.float32),
                          valid=tp["valid"][i], name=str(tp["names"][i])))
    print(f"cases: {sum(c['kind'][0] == 'A' for c in cases)} synthetic, {sum(c['kind'][0] == 'B' for c in cases)} real-cloud")

    rows, feats, shows = [], {}, []
    for ci, c in enumerate(cases):
        region = (c["mask"] > 0) & c["valid"]
        if region.sum() < 200:
            continue
        km = fit_landcover(c["truth"])
        outs = {}
        for t, m in models.items():
            out = c["cloudy"] if m is None else reconstruct_scene(m, c["cloudy"], c["mask"], c["ref"], norm, dev,
                                                                   tta=True)[0]
            outs[t] = out
            rows.append({"kind": c["kind"], "case": ci, "model": t, **evaluate(out, c["truth"], region, km),
                         "sharpness_ratio": sharpness_ratio(out, c["truth"], region)})
            feats.setdefault((c["kind"], t), []).append(patch_features(out, region))
        if c["kind"].startswith("B") and len(shows) < 5:
            shows.append((c, outs, region))
        if ci % 10 == 0:
            print(f"  {ci + 1}/{len(cases)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(P["results"] / "liss4_per_case.csv", index=False)
    cols = ["psnr_cloud", "ssim_cloud", "mae_cloud", "sam_cloud", "ndvi_mae_cloud", "lulc_agree_cloud", "sharpness_ratio"]
    summ = df.groupby(["kind", "model"], sort=False)[cols].mean().reset_index()
    summ["niqe_cloud"] = [niqe.score(np.concatenate(feats[(r.kind, r.model)])) for r in summ.itertuples()]
    summ["n"] = [int((df["kind"] == r.kind).sum() / len(models)) for r in summ.itertuples()]
    summ["model_name"] = summ["model"].map(LABELS)
    summ.to_csv(P["results"] / "liss4_summary.csv", index=False)
    with open(P["results"] / "liss4_summary.md", "w") as f:
        f.write("Real LISS-IV (Bhoonidhi, Resourcesat-2/2A), east test area never used for training.\n")
        for k, g in summ.groupby("kind", sort=False):
            f.write(f"\n### {k} ({g['n'].iloc[0]} cases)\n\n| Model | PSNR cloud | SSIM | MAE | SAM ° | NDVI MAE | Land-cover | "
                    f"Sharpness | NIQE ↓ |\n|---|---|---|---|---|---|---|---|---|\n")
            for _, r in g.iterrows():
                f.write(f"| {r.model_name} | {r.psnr_cloud:.2f} | {r.ssim_cloud:.3f} | {r.mae_cloud:.4f} | {r.sam_cloud:.2f} | "
                        f"{r.ndvi_mae_cloud:.3f} | {100 * r.lulc_agree_cloud:.1f}% | {r.sharpness_ratio:.2f} | {r.niqe_cloud:.2f} |\n")
    print((P["results"] / "liss4_summary.md").read_text())

    if shows:
        show_models = [t for t in ("classical", "meghgan_real", "meghgan_liss4") if t in models]
        cols_ = ["Real cloudy LISS-IV (11 May 2024)", "Deep mask", "Reference (13 Nov 2023, matched)"] + \
                [LABELS[t].split(" (")[0] for t in show_models] + ["Truth: 23 May 2024"]
        fig, ax = plt.subplots(len(shows), len(cols_), figsize=(3.2 * len(cols_), 3.3 * len(shows)), squeeze=False)
        for r_, (c, outs, _) in enumerate(shows):
            lim = display_limits(c["truth"])
            ims = [to_display(c["cloudy"]), c["mask"], to_display(match_reference(c["cloudy"], c["mask"], c["ref"]), limits=lim)]
            ims += [to_display(outs[t], limits=lim) for t in show_models] + [to_display(c["truth"], limits=lim)]
            for cc, im in enumerate(ims):
                ax[r_, cc].imshow(im, **({"cmap": "viridis", "vmin": 0, "vmax": 2, "interpolation": "nearest"} if cc == 1 else {}))
                ax[r_, cc].set_xticks([]), ax[r_, cc].set_yticks([])
                if r_ == 0:
                    ax[r_, cc].set_title(cols_[cc], fontsize=9)
        plt.tight_layout()
        plt.savefig(P["results"] / "figures" / "liss4_real_pairs.png", dpi=95)
    (P["results"] / "liss4_eval_info.json").write_text(json.dumps({"synthetic_cases": N_SYNTH, "models": list(models)}))


if __name__ == "__main__":
    main()
