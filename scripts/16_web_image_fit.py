import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.io import load_config
from meghmukt.webimage import DARK_QUANTILE, NirFromRGB, dark_levels


def main():
    cfg = load_config()
    P = cfg["paths"]
    spec = importlib.util.spec_from_file_location("bb", ROOT / "scripts" / "09_blue_band.py")
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)
    rng = np.random.default_rng(0)
    Xtr, Btr = bb.load_pixels(cfg, cfg["split"]["train"], 20000, rng)
    Xte, Bte = bb.load_pixels(cfg, cfg["split"]["test"] + cfg["split"]["val"], 20000, rng)

    bgrn = np.stack([Btr, Xtr[0], Xtr[1], Xtr[2]])
    levels = {k: dark_levels(v, bgrn.mean(0)) for k, v in zip(("blue", "green", "red", "nir"), bgrn)}

    from meghmukt.io import Harmoniser
    harm = Harmoniser.load(P["processed"] / "harmoniser.json")
    spectra = np.load(P["processed"] / "cloud_spectra.npy").T[:, :, None]
    cg, cr, cn = np.median(harm.inverse(spectra)[..., 0], 1)
    cloud_levels = {"blue": float(np.mean([cg, cr, cn])), "green": float(cg), "red": float(cr), "nir": float(cn)}

    model = NirFromRGB.fit(Btr, Xtr[0], Xtr[1], Xtr[2])
    pred, true = model.predict(Bte, Xte[0], Xte[1]), Xte[2]
    r2 = float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))
    nd_t = (true - Xte[1]) / (true + Xte[1] + 1e-6)
    nd_p = (pred - Xte[1]) / (pred + Xte[1] + 1e-6)
    ndvi_r2 = float(1 - np.sum((nd_p - nd_t) ** 2) / np.sum((nd_t - nd_t.mean()) ** 2))
    out = {"levels": levels, "cloud_levels": cloud_levels, "dark_quantile": DARK_QUANTILE, "nir_model": model.params,
           "nir_r2_unseen": r2, "ndvi_r2_unseen": ndvi_r2, "ndvi_mae_unseen": float(np.abs(nd_p - nd_t).mean())}
    (P["processed"] / "web_image.json").write_text(json.dumps(out))
    (P["results"] / "nir_from_rgb_study.json").write_text(json.dumps(
        {k: v for k, v in out.items() if k not in ("nir_model", "levels", "cloud_levels")} |
        {"train_areas": cfg["split"]["train"], "test_areas": cfg["split"]["test"] + cfg["split"]["val"]}, indent=2))
    print(f"[web] NIR from RGB on unseen areas: R² {r2:.3f}  (NDVI R² {ndvi_r2:.3f}, NDVI MAE {out['ndvi_mae_unseen']:.3f})")
    print("[web] saved data/processed/web_image.json")


if __name__ == "__main__":
    main()
