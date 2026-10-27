import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import FixedSet, Normaliser
from meghmukt.inference import confidence_from, depth_factor, predict, uncertainty_map
from meghmukt.io import display_limits, get_device, load_config, to_display
from meghmukt.metrics import evaluate, fit_landcover, uncertainty_quality
from meghmukt.models import MODEL_NAMES, ClassicalFill, load_checkpoint, soft_mask
from meghmukt.perception import NIQE, patch_features, sharpness_ratio

LABELS = {"copy_input": "Do nothing (cloudy input)", **MODEL_NAMES,
          "meghgan_nospectral": "M2 MeghGAN w/o spectral loss", "meghgan_liss4ft": "M2 MeghGAN + LISS-IV fine-tune",
          "meghgan_noadv": "M2 without GAN loss (ablation)", "meghgan_charb": "M2 + Charbonnier loss",
          "meghgan_real": "M2 MeghGAN + real-cloud training", "diffusion_real": "M3 Diffusion + real-cloud training",
          "unet_liss4ft": "M1 U-Net + LISS-IV fine-tune"}


def load_models(ckdir, device):
    models = {"copy_input": None, "classical": ClassicalFill()}
    for tag in ("unet", "unet_liss4ft", "meghgan", "meghgan_nospectral", "meghgan_noadv", "meghgan_charb",
                "meghgan_real", "meghgan_liss4ft", "diffusion", "diffusion_real"):
        p = ckdir / f"{tag}.pt"
        if p.exists():
            models[tag] = load_checkpoint(p, device)[0]
    return models


def run_model(model, cond, device, batch=16):
    preds, uncs = [], []
    for i in range(0, len(cond), batch):
        c = cond[i:i + batch].to(device)
        if model is None:
            m, s = c[:, :3], torch.zeros_like(c[:, :3])
        else:
            m, s = predict(model, c, tta=True, n_samples=4)
        preds.append(m.cpu()), uncs.append(s.cpu())
    return torch.cat(preds), torch.cat(uncs)


def main():
    cfg = load_config()
    P = cfg["paths"]
    device = get_device(cfg["train"]["device"])
    norm = Normaliser.load(P["processed"] / "norm.json")
    models = load_models(P["checkpoints"], device)
    fig_dir = P["results"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("models:", list(models))

    rows, examples, conf_bins = [], {}, []
    niqe = NIQE.load(P["processed"] / "niqe_pristine.npz")
    perception = {}
    for set_name in ("test", "liss4"):
        ds = FixedSet(P["processed"] / f"eval_{set_name}.npz")
        cond_all = torch.from_numpy(ds.cond.astype(np.float32))
        gt = norm.inverse(ds.gt.astype(np.float32))
        region = (soft_mask(cond_all, blur=0) > 0.5).numpy()[:, 0]
        kms = [fit_landcover(g) for g in gt]
        for scenario in ("with_ref", "no_ref"):
            cond = cond_all.clone()
            if scenario == "no_ref":
                cond[:, 5:9] = 0
            for tag, model in models.items():
                if model is None and scenario == "no_ref":
                    continue
                if tag.endswith("liss4ft") and set_name != "liss4":
                    continue
                pred, unc = run_model(model, cond, device)
                pred = np.clip(norm.inverse(pred.numpy()), 0, 1)
                unc = (unc.numpy() * norm.std).mean(1)
                for i in range(len(pred)):
                    rows.append({"set": set_name, "scenario": scenario, "model": tag, "sample": i,
                                 **evaluate(pred[i], gt[i], region[i], kms[i]),
                                 "sharpness_ratio": sharpness_ratio(pred[i], gt[i], region[i])})
                perception[(set_name, scenario, tag)] = niqe.score(
                    np.concatenate([patch_features(pred[i], region[i]) for i in range(len(pred))]))
                examples[(set_name, scenario, tag)] = (pred, unc)
                cur = [r for r in rows if r["set"] == set_name and r["scenario"] == scenario and r["model"] == tag]
                print(f"{set_name:6s} {scenario:8s} {tag:20s} PSNR cloud {np.mean([r['psnr_cloud'] for r in cur]):6.2f} dB"
                      f"  SSIM cloud {np.mean([r['ssim_cloud'] for r in cur]):.3f}  SAM cloud {np.mean([r['sam_cloud'] for r in cur]):5.2f}°",
                      flush=True)

            if (set_name, scenario, "diffusion") not in examples:
                continue
            spread = examples[(set_name, scenario, "diffusion")][1]
            deployed = np.stack([uncertainty_map(region[i], spread[i]) for i in range(len(region))])
            examples[(set_name, scenario, "deployed_unc")] = deployed
            for r in rows:
                if r["set"] == set_name and r["scenario"] == scenario and r["model"] not in ("copy_input", "classical"):
                    p_ = examples[(set_name, scenario, r["model"])][0][r["sample"]]
                    r.update(uncertainty_quality(deployed[r["sample"]], np.abs(p_ - gt[r["sample"]]).mean(0),
                                                 region[r["sample"]]))
            if set_name == "test" and scenario == "with_ref":
                err = np.abs(examples[(set_name, scenario, "meghgan")][0] - gt).mean(1)
                cands = {"MeghGAN test-time-augmentation spread": examples[(set_name, scenario, "meghgan")][1],
                         "diffusion-sample spread": spread,
                         "depth inside cloud": np.stack([depth_factor(m) for m in region]),
                         "deployed: diffusion spread x depth": deployed}
                study = {}
                for name, u in cands.items():
                    q = uncertainty_quality(u[region], err[region], np.ones(region.sum(), bool))
                    study[name] = {k: round(v, 3) for k, v in q.items()}
                    e_, u_ = err[region], u[region]
                    edges = np.quantile(u_, np.linspace(0, 1, 11))
                    b = np.clip(np.searchsorted(edges, u_, side="right") - 1, 0, 9)
                    for k in range(10):
                        conf_bins.append({"estimator": name, "bin": 10 - k, "mean_abs_error": float(e_[b == k].mean())})
                (P["results"] / "uncertainty_study.json").write_text(json.dumps(
                    {"question": "which per-pixel uncertainty predicts MeghGAN's real error? (unseen test areas)",
                     "metric": "unc_spearman = rank correlation; err_ratio = error on least- vs most-confident 20 %",
                     "results": study}, indent=2))
                print("uncertainty study:", json.dumps(study))
        examples[(set_name, "inputs")] = (norm.inverse(cond_all[:, :3].numpy()), norm.inverse(cond_all[:, 5:8].numpy()),
                                          region, gt)
        perception[(set_name, "truth", "ground_truth")] = niqe.score(
            np.concatenate([patch_features(gt[i], region[i]) for i in range(len(gt))]))

    pd.DataFrame(conf_bins).sort_values(["estimator", "bin"]).to_csv(P["results"] / "confidence_bins.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(P["results"] / "metrics_per_sample.csv", index=False)
    cols = ["psnr_cloud", "ssim_cloud", "mae_cloud", "rmse_cloud", "sam_cloud", "ndvi_mae_cloud", "lulc_agree_cloud",
            "psnr_all", "ssim_all", "psnr_clear", "sharpness_ratio", "unc_spearman", "err_ratio_low_vs_high_conf"]
    summary = df.groupby(["set", "scenario", "model"], sort=False)[[c for c in cols if c in df]].mean().reset_index()
    summary["niqe_cloud"] = [perception.get((r.set, r.scenario, r.model), np.nan) for r in summary.itertuples()]
    summary["model_name"] = summary["model"].map(LABELS)
    (P["results"] / "perception.json").write_text(json.dumps(
        {"niqe_ground_truth": {s_: round(v, 3) for (s_, sc, t), v in perception.items() if t == "ground_truth"},
         "note": "NIQE fitted on cloud-free NE-India imagery (train areas); lower = more natural"}, indent=2))
    summary.to_csv(P["results"] / "summary.csv", index=False)
    with open(P["results"] / "summary.md", "w") as f:
        for (s, sc), g in summary.groupby(["set", "scenario"], sort=False):
            f.write(f"\n### {s} — {sc}\n\n| Model | PSNR cloud (dB) | SSIM cloud | MAE cloud | SAM cloud (°) | NDVI MAE | Land-cover agree | Sharpness | NIQE ↓ | Unc. ρ |\n|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in g.iterrows():
                f.write(f"| {r.model_name} | {r.psnr_cloud:.2f} | {r.ssim_cloud:.3f} | {r.mae_cloud:.4f} | {r.sam_cloud:.2f} | "
                        f"{r.ndvi_mae_cloud:.3f} | {100 * r.lulc_agree_cloud:.1f}% | {r.sharpness_ratio:.2f} | {r.niqe_cloud:.2f} | "
                        f"{'' if pd.isna(r.get('unc_spearman')) else f'{r.unc_spearman:.2f}'} |\n")
    print((P["results"] / "summary.md").read_text())

    for set_name, n_show in (("test", 5), ("liss4", 4)):
        cloudy, ref, region, gt = examples[(set_name, "inputs")]
        tags = [t for t in ("classical", "unet", "meghgan", "diffusion", "meghgan_liss4ft") if (set_name, "with_ref", t) in examples]
        idx = np.argsort([-r for r in region.mean((1, 2))])[:: max(1, len(region) // n_show)][:n_show]
        cols_ = ["Cloudy input", "Temporal ref."] + [LABELS[t].split(" (")[0] for t in tags] + ["Ground truth", "Confidence map"]
        fig, ax = plt.subplots(len(idx), len(cols_), figsize=(2.3 * len(cols_), 2.4 * len(idx)))
        for r_, i in enumerate(idx):
            lim = display_limits(gt[i])
            ims = [to_display(cloudy[i], limits=lim), to_display(ref[i], limits=lim)]
            ims += [to_display(examples[(set_name, "with_ref", t)][0][i], limits=lim) for t in tags]
            ims += [to_display(gt[i], limits=lim)]
            for c_, im in enumerate(ims):
                ax[r_, c_].imshow(im)
            u = examples[(set_name, "with_ref", "deployed_unc")][i]
            ax[r_, -1].imshow(confidence_from(u, region[i]), cmap="RdYlGn", vmin=0, vmax=1)
            for c_ in range(len(cols_)):
                ax[r_, c_].set_xticks([]), ax[r_, c_].set_yticks([])
                if r_ == 0:
                    ax[r_, c_].set_title(cols_[c_], fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / f"compare_{set_name}.png", dpi=110)
        plt.close()
    (P["results"] / "eval_info.json").write_text(json.dumps({"device": str(device), "models": list(models)}, indent=2))
    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
