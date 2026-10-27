import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.io import load_config

COL = {"copy_input": "#8a8984", "classical": "#2a78d6", "unet": "#eb6834", "meghgan": "#1baf7a",
       "meghgan_nospectral": "#4a3aa7", "diffusion": "#e87ba4", "meghgan_liss4ft": "#008300", "unet_liss4ft": "#eda100",
       "meghgan_noadv": "#52514e", "meghgan_charb": "#c3c2b7", "meghgan_real": "#0a7a52", "diffusion_real": "#b0306a"}
NAME = {"copy_input": "Do nothing", "classical": "M0 Classical", "unet": "M1 U-Net", "meghgan": "M2 MeghGAN (ours)",
        "meghgan_nospectral": "M2 w/o spectral loss", "diffusion": "M3 Diffusion",
        "meghgan_liss4ft": "M2 + LISS-IV fine-tune", "unet_liss4ft": "M1 + LISS-IV fine-tune",
        "meghgan_noadv": "M2 without GAN loss", "meghgan_charb": "M2 + Charbonnier",
        "meghgan_real": "M2 + real-cloud training", "diffusion_real": "M3 + real-cloud training"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": "white"})


def training_curves(P, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for tag in ("unet", "meghgan", "meghgan_nospectral", "diffusion"):
        f = P["results"] / "train_logs" / f"{tag}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        ax.plot(d["step"], d["val_psnr_cloud"], color=COL[tag], lw=2, marker="o", ms=4, label=NAME[tag])
        ax.annotate(f"{NAME[tag]}  {d['val_psnr_cloud'].max():.1f} dB", (d["step"].iloc[-1], d["val_psnr_cloud"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("training step"), ax.set_ylabel("validation PSNR, cloud pixels (dB)")
    ax.set_title("Training progress (validation area: Silchar)", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8), ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.28)
    plt.tight_layout(), plt.savefig(out / "training_curves.png", dpi=150), plt.close()


def comparison(S, out, set_name="test"):
    g = S[(S.set == set_name) & (S.scenario == "with_ref")]
    g = g[~g.model.str.endswith("liss4ft")]
    metrics = [("psnr_cloud", "PSNR, cloud pixels (dB) ↑", "{:.2f}"), ("ssim_cloud", "SSIM, cloud pixels ↑", "{:.3f}"),
               ("sam_cloud", "Spectral angle SAM (°) ↓", "{:.2f}"), ("ndvi_mae_cloud", "NDVI error ↓", "{:.3f}")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    y = np.arange(len(g))[::-1]
    for ax, (m, title, fmt) in zip(axs, metrics):
        vals = g[m].values
        ax.barh(y, vals, color=[COL[t] for t in g.model], height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, vals.max() * 1.25), ax.grid(axis="x", color=GRID, lw=0.8), ax.set_axisbelow(True)
    axs[0].set_yticks(y, [NAME[t] for t in g.model])
    fig.suptitle("Unseen test areas (Agartala, Imphal) — with temporal reference", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout(), plt.savefig(out / "comparison_metrics.png", dpi=150), plt.close()


def reference_ablation(S, out):
    g = S[S.set == "test"]
    tags = [t for t in ("classical", "unet", "meghgan", "diffusion") if t in set(g.model)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, t in enumerate(tags):
        w = g[(g.model == t) & (g.scenario == "with_ref")].psnr_cloud.values[0]
        n = g[(g.model == t) & (g.scenario == "no_ref")].psnr_cloud.values[0]
        ax.plot([n, w], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([n], [i], s=70, facecolor="white", edgecolor=COL[t], lw=2, zorder=2)
        ax.scatter([w], [i], s=70, color=COL[t], zorder=3)
        label = "no reference input" if t == "unet" else f"{w - n:+.1f} dB"
        ax.text(max(n, w) + 0.3, i, label, va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(tags)), [NAME[t] for t in tags])
    ax.set_xlabel("PSNR on cloud pixels (dB)   ○ without reference   ● with reference")
    ax.set_title("Value of the temporal reference", loc="left"), ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlim(right=ax.get_xlim()[1] + 2)
    plt.tight_layout(), plt.savefig(out / "reference_ablation.png", dpi=150), plt.close()


def confidence_check(S, D, out):
    f = D / "confidence_bins.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    col = {"MeghGAN test-time-augmentation spread": "#8a8984", "depth inside cloud": "#2a78d6",
           "diffusion-sample spread": "#e87ba4", "deployed: diffusion spread x depth": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    for est, g in d.groupby("estimator"):
        ax.plot(g["bin"], g["mean_abs_error"], color=col.get(est, MUTED), lw=2.6 if est.startswith("deployed") else 1.6,
                marker="o", ms=4, label=est)
    ax.set_xlabel("confidence decile (1 = least confident → 10 = most confident)")
    ax.set_ylabel("MeghGAN mean |error| (normalised DN)")
    ax.set_title("Does the confidence map point at the real errors?", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8), ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout(), plt.savefig(out / "confidence_check.png", dpi=150), plt.close()


def main():
    P = load_config()["paths"]
    out = P["results"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    training_curves(P, out)
    S = pd.read_csv(P["results"] / "summary.csv")
    comparison(S, out)
    reference_ablation(S, out)
    confidence_check(S, P["results"], out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
