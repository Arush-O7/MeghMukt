import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meghmukt.io import load_config, write_raster

spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "01_download_sentinel2.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

AOIS = {
    "agartala": (91.28, 23.83), "imphal": (93.94, 24.80), "silchar": (92.80, 24.82),
    "tezpur": (92.80, 26.63), "itanagar": (93.62, 27.08), "tura": (90.22, 25.51),
    "kohima": (94.11, 25.67), "dhubri": (89.98, 26.02), "lakhimpur": (94.10, 27.23),
}
TRAIN_AOIS = {
    "guwahati": (91.60, 26.12), "majuli": (94.20, 26.95), "kaziranga": (93.35, 26.62), "shillong": (91.88, 25.57),
    "dimapur": (93.73, 25.90), "pasighat": (95.33, 28.07), "aizawl": (92.72, 23.73),
}
CHIP = 768
MAX_GAP_DAYS = 10
PAIRS_PER_AOI = 3 if "--train" in sys.argv else 2


def days(a, b):
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def save(it, lon, lat, path, kind):
    epsg, bounds = dl.chip_bounds(it, lon, lat, CHIP * 10)
    bands = []
    for key in ("green", "red", "nir"):
        arr, crs = dl.read_window(it["assets"][key]["href"], bounds, CHIP)
        bands.append(arr)
    scl, _ = dl.read_window(it["assets"]["scl"]["href"], bounds, CHIP)
    so = [dl.scale_offset(it, k) for k in ("green", "red", "nir")]
    prof = dict(driver="GTiff", width=CHIP, height=CHIP, crs=crs,
                transform=rasterio.transform.from_bounds(*bounds, CHIP, CHIP))
    write_raster(path, np.stack(bands + [scl]).astype(np.uint16), prof,
                 tags={"source": it["id"], "scale_offset": json.dumps(so), "kind": kind},
                 band_names=["green_B03", "red_B04", "nir_B08", "SCL"])


def main():
    cfg = load_config()
    cfg["sentinel2"]["date_range"] = "2022-10-01T00:00:00Z/2025-05-31T23:59:59Z"
    train = "--train" in sys.argv
    out_root = cfg["paths"]["raw"] / ("real_pairs_train" if train else "real_pairs")
    for name, (lon, lat) in (TRAIN_AOIS if train else AOIS).items():
        out = out_root / name
        if (out / "meta.json").exists() and "--refresh-reference" not in sys.argv:
            print(f"[{name}] done already")
            continue
        out.mkdir(parents=True, exist_ok=True)
        by_date = {}
        for it in sorted(dl.search(cfg, lon, lat), key=lambda f: f["properties"]["eo:cloud_cover"]):
            by_date.setdefault(it["properties"]["datetime"][:10], it)
        scored = []
        for d, it in sorted(by_date.items()):
            _, b = dl.chip_bounds(it, lon, lat, CHIP * 10)
            try:
                scl, _ = dl.read_window(it["assets"]["scl"]["href"], b, CHIP // 4)
            except Exception:
                continue
            if np.isin(scl, dl.SCL_NODATA).mean() > 0.01:
                continue
            scored.append(dict(date=d, item=it, cloud=float(np.isin(scl, dl.SCL_CLOUD).mean()),
                               shadow=float(np.isin(scl, dl.SCL_SHADOW).mean())))
        clear = [s for s in scored if s["cloud"] + s["shadow"] < 0.01]
        cloudy = [s for s in scored if 0.10 <= s["cloud"] <= 0.60]
        pairs, used = [], set()
        for c in sorted(cloudy, key=lambda s: abs(s["cloud"] - 0.35)):
            near = [k for k in clear if days(k["date"], c["date"]) <= MAX_GAP_DAYS and k["date"] not in used]
            if not near or c["date"] in used:
                continue
            k = min(near, key=lambda k: days(k["date"], c["date"]))
            pairs.append((c, k)), used.update({c["date"], k["date"]})
            if len(pairs) == PAIRS_PER_AOI:
                break
        if not pairs:
            print(f"[{name}] no near-date pair found ({len(scored)} usable dates)")
            (out / "meta.json").write_text(json.dumps({"aoi": name, "pairs": []}))
            continue
        far = [k for k in clear if all(days(k["date"], d) >= 30 for d in used)]
        ref = min(far, key=lambda k: min(days(k["date"], d) for d in used)) if far else None
        meta = {"aoi": name, "lon": lon, "lat": lat, "pairs": [], "reference": None}
        for c, k in pairs:
            if not (out / f"{c['date']}_cloudy.tif").exists():
                save(c["item"], lon, lat, out / f"{c['date']}_cloudy.tif", "cloudy")
            if not (out / f"{k['date']}_truth.tif").exists():
                save(k["item"], lon, lat, out / f"{k['date']}_truth.tif", "truth")
            meta["pairs"].append({"cloudy": f"{c['date']}_cloudy.tif", "truth": f"{k['date']}_truth.tif",
                                  "gap_days": days(c["date"], k["date"]), "cloud": c["cloud"], "shadow": c["shadow"]})
            print(f"[{name}] pair {c['date']} (cloud {c['cloud']:.0%}) <-> {k['date']}  gap {days(c['date'], k['date'])} d")
        for old_ref in out.glob("*_reference.tif"):
            old_ref.unlink()
        if ref:
            save(ref["item"], lon, lat, out / f"{ref['date']}_reference.tif", "reference")
            meta["reference"] = f"{ref['date']}_reference.tif"
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    total = sum(f.stat().st_size for f in out_root.rglob("*.tif")) / 1e6
    print(f"Done. real-pair chips on disk: {total:.0f} MB")


if __name__ == "__main__":
    main()
