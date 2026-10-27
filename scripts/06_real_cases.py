import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import Normaliser
from meghmukt.inference import export_analysis_ready, reconstruct_scene
from meghmukt.io import Harmoniser, display_limits, get_device, load_config, read_raster, to_display
from meghmukt.masking import CLOUD, MaskParams, coverage, detect_auto, scl_to_mask
from meghmukt.models import CONFIDENCE_MODEL, DEFAULT_MODEL, DEFAULT_MODEL_S2, load_checkpoint
from meghmukt.reference import fetch_sentinel2_reference, to_liss4_scale


def panel(ax_row, title, cloudy, mask, ref, out, conf):
    lim = display_limits(out)
    items = [("Real cloudy input", to_display(cloudy)),
             ("Detected cloud / shadow", mask),
             ("Temporal reference", to_display(ref, limits=lim) if ref is not None else None),
             ("MeghMukt output", to_display(out, limits=lim)), ("Confidence", conf)]
    for a, (t, im) in zip(ax_row, items):
        a.set_xticks([]), a.set_yticks([])
        if im is None:
            a.text(0.5, 0.5, "no reference\navailable", ha="center", va="center", transform=a.transAxes)
        elif t.startswith("Detected"):
            a.imshow(im, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        elif t == "Confidence":
            a.imshow(im, cmap="RdYlGn", vmin=0, vmax=1)
        else:
            a.imshow(im)
        a.set_title(t, fontsize=9)
    ax_row[0].set_ylabel(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_S2, help="model used for the Sentinel-2 cases")
    ap.add_argument("--liss4-model", default=DEFAULT_MODEL, help="model used for the LISS-IV cases")
    ap.add_argument("--mask-method", default="deep", choices=["deep", "rules"])
    a = ap.parse_args()
    cfg = load_config()
    P = cfg["paths"]
    device = get_device(cfg["train"]["device"])
    norm = Normaliser.load(P["processed"] / "norm.json")
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    mp = MaskParams(**json.loads((P["processed"] / "mask_params.json").read_text())["params"])
    model = load_checkpoint(P["checkpoints"] / f"{a.model}.pt", device)[0]
    lck = P["checkpoints"] / f"{a.liss4_model}.pt"
    lmodel = load_checkpoint(lck, device)[0] if lck.exists() else model
    conf_model = load_checkpoint(P["checkpoints"] / f"{CONFIDENCE_MODEL}.pt", device)[0]
    out_dir = P["results"] / "real_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, report = [], {}

    def run(name, m, cloudy, ref, profile=None, mask=None, pixel_m=10.0):
        mask = detect_auto(cloudy, a.mask_method, pixel_m, mp)[0] if mask is None else mask
        out, unc, conf = reconstruct_scene(m, cloudy, mask, ref, norm, device, cfg["inference"]["tile"],
                                           cfg["inference"]["overlap"], tta=True, conf_model=conf_model)
        cases.append((name, cloudy, mask, ref, out, conf))
        region = mask > 0
        report[name] = {**coverage(mask), "reference": ref is not None,
                        "mean_conf_reconstructed": float(conf[region].mean()) if region.any() else 1.0}
        if profile is not None:
            export_analysis_ready(out_dir / f"{name}.tif", out, mask, conf, profile, a.model)
        print(f"{name}: {report[name]}")
        return mask

    tile_dn, tprof = read_raster(P["liss4_samples"] / "R2F13JUN2024068249010700_cloudy.tif")
    tile = to_liss4_scale(tile_dn, "liss4", harm)
    ref_a, info = fetch_sentinel2_reference(tprof, harm)
    report["reference_A"] = info
    run("A_liss4_scene_S2ref", lmodel, tile, ref_a, tprof, pixel_m=5.0)

    patch = np.load(P["processed"] / "liss4.npz")["cloudy_patch"]
    run("B_liss4_patch_noref", lmodel, patch, None, pixel_m=5.0)

    for aoi in cfg["split"]["test"]:
        d = np.load(P["processed"] / f"s2_{aoi}.npz")
        for k in range(len(d["cloudy"])):
            x = d["cloudy"][k].astype(np.float32)
            m = run(f"C_{aoi}_{d['dates_cloudy'][k]}", model, x, d["clear"][-1].astype(np.float32))
            scl = scl_to_mask(d["scl"][k]) == CLOUD
            det = m == CLOUD
            report[f"C_{aoi}_{d['dates_cloudy'][k]}"]["mask_iou_vs_SCL"] = float((scl & det).sum() / max((scl | det).sum(), 1))

    s2_30m = P["s2_30m"]
    if (s2_30m / "real_cloudy.tif").exists():
        rc, bprof = read_raster(s2_30m / "real_cloudy.tif")
        cr, _ = read_raster(s2_30m / "clean_reference.tif")
        run("D_s2_30m", model, to_liss4_scale(rc[1:4], "s2", harm), to_liss4_scale(cr[1:4], "s2", harm), bprof, pixel_m=30.0)

    groups = {"liss4": [c for c in cases if c[0][0] in "AB"], "sentinel2": [c for c in cases if c[0][0] in "CD"]}
    for g, cs in groups.items():
        if not cs:
            continue
        fig, ax = plt.subplots(len(cs), 5, figsize=(15, 3.2 * len(cs)), squeeze=False)
        for row, c in zip(ax, cs):
            panel(row, c[0], *c[1:])
        plt.tight_layout()
        plt.savefig(out_dir / f"real_{g}.png", dpi=110)
        plt.close()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("->", out_dir)


if __name__ == "__main__":
    main()
