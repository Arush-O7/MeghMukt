import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.dataset import Normaliser
from meghmukt.inference import export_analysis_ready, reconstruct_scene
from meghmukt.io import (Harmoniser, display_limits, get_device, load_config, read_raster, to_display,
                         write_raster)
from meghmukt.masking import MaskParams, coverage, detect_auto, pixel_size
from meghmukt.models import CONFIDENCE_MODEL, MODEL_NAMES, ClassicalFill, default_model, load_checkpoint
from meghmukt.reference import fetch_sentinel2_reference, load_reference, to_liss4_scale


def load_mask_params(cfg):
    p = cfg["paths"]["processed"] / "mask_params.json"
    return MaskParams(**json.loads(p.read_text())["params"]) if p.exists() else MaskParams()


def quicklook(path, cloudy, mask, ref, out, conf, title):
    lim = display_limits(out)
    panels = [("Cloudy input", to_display(cloudy, limits=lim)), ("Cloud / shadow mask", mask)]
    if ref is not None:
        panels.append(("Temporal reference", to_display(ref, limits=lim)))
    panels += [("MeghMukt output", to_display(out, limits=lim)), ("Confidence", conf)]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    for a, (t, im) in zip(ax, panels):
        if t.startswith("Cloud /"):
            a.imshow(im, cmap="viridis", vmin=0, vmax=2)
        elif t == "Confidence":
            a.imshow(im, cmap="RdYlGn", vmin=0, vmax=1)
        else:
            a.imshow(im)
        a.set_title(t), a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def run(input_path, out_path, sensor="liss4", bands=(1, 2, 3), reference=None, ref_sensor="s2", ref_bands=(1, 2, 3),
        fetch_reference=False, model_tag=None, mask_path=None, tta=True, confidence=True, natural=False,
        mask_method="deep", log=print):
    cfg = load_config()
    P = cfg["paths"]
    device = get_device(cfg["train"]["device"])
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    norm = Normaliser.load(P["processed"] / "norm.json")
    t0 = time.time()

    arr, profile = read_raster(input_path)
    cloudy = to_liss4_scale(arr[[b - 1 for b in bands]], sensor, harm)
    log(f"input {Path(input_path).name}: {cloudy.shape[1]}x{cloudy.shape[2]} px, sensor={sensor}")

    if mask_path:
        mask = read_raster(mask_path)[0][0].astype(np.uint8)
    else:
        mask, _ = detect_auto(cloudy, mask_method, pixel_size(profile), load_mask_params(cfg))
    cov = coverage(mask)
    log(f"mask ({'file' if mask_path else mask_method}): cloud {cov['cloud']:.1%}, shadow {cov['shadow']:.1%}")

    ref, ref_info = None, {}
    if reference:
        ref = load_reference(reference, ref_sensor, profile, harm, [b - 1 for b in ref_bands])
        ref_info = {"reference": str(reference)}
    elif fetch_reference:
        log("searching Sentinel-2 archive for a clear reference over this footprint ...")
        ref, ref_info = fetch_sentinel2_reference(profile, harm, log=log)
        log(f"reference: {ref_info}")

    model_tag = model_tag or default_model(sensor)
    if model_tag == "classical":
        model = ClassicalFill()
    else:
        ck = P["checkpoints"] / f"{model_tag}.pt"
        model = load_checkpoint(ck, device)[0]
    dck = P["checkpoints"] / f"{CONFIDENCE_MODEL}.pt"
    conf_model = load_checkpoint(dck, device)[0] if (confidence and dck.exists()) else None
    out, unc, conf = reconstruct_scene(model, cloudy, mask, ref, norm, device, cfg["inference"]["tile"],
                                       cfg["inference"]["overlap"], tta=tta, conf_model=conf_model)
    name = MODEL_NAMES.get(model_tag, model_tag)
    region = mask > 0
    report = {"input": str(input_path), "model": model_tag, "seconds": round(time.time() - t0, 1), **cov,
              "mean_confidence_reconstructed": float(conf[region].mean()) if region.any() else 1.0,
              "low_confidence_fraction": float((conf[region] < 0.5).mean()) if region.any() else 0.0, **ref_info}
    export_analysis_ready(out_path, out, mask, conf, profile, name,
                          {k.upper(): v for k, v in report.items() if k not in ("input",)})
    quicklook(Path(out_path).with_suffix(".png"), cloudy, mask, ref, out, conf, f"{Path(input_path).name} — {name}")
    if natural:
        rgb = to_display(out, "natural").transpose(2, 0, 1)
        prof = {k: v for k, v in profile.items() if k in ("driver", "crs", "transform", "width", "height")}
        write_raster(str(out_path).replace(".tif", "_natural.tif"), rgb, prof,
                     tags={"PRODUCT": "MeghMukt natural colour (blue band synthesised)"}, band_names=["red", "green", "blue_synthetic"])
    Path(out_path).with_suffix(".json").write_text(json.dumps(report, indent=2))
    log(f"done in {report['seconds']} s -> {out_path}")
    return out, mask, conf, ref, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out")
    ap.add_argument("--sensor", default="liss4", choices=["liss4", "s2"])
    ap.add_argument("--bands", type=int, nargs=3, default=[1, 2, 3], help="1-based Green Red NIR band numbers")
    ap.add_argument("--reference")
    ap.add_argument("--ref-sensor", default="s2", choices=["liss4", "s2"])
    ap.add_argument("--ref-bands", type=int, nargs=3, default=[1, 2, 3])
    ap.add_argument("--fetch-reference", action="store_true")
    ap.add_argument("--model", help="default: the model trained on real data of the input's sensor")
    ap.add_argument("--mask", help="use this mask GeoTIFF (0 clear, 1 cloud, 2 shadow) instead of detecting one")
    ap.add_argument("--mask-method", default="deep", choices=["deep", "rules"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-confidence", action="store_true", help="skip the diffusion-based confidence map (faster)")
    ap.add_argument("--natural-colour", action="store_true", help="also write a natural-colour RGB GeoTIFF")
    a = ap.parse_args()
    tag = a.model or default_model(a.sensor)
    out = a.out or str(load_config()["paths"]["results"] / "scenes" / (Path(a.input).stem + f"_{tag}_cloudfree.tif"))
    run(a.input, out, a.sensor, a.bands, a.reference, a.ref_sensor, a.ref_bands, a.fetch_reference, tag, a.mask,
        not a.no_tta, not a.no_confidence, a.natural_colour, a.mask_method)


if __name__ == "__main__":
    main()
