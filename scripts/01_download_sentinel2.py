import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meghmukt.io import load_config, write_raster

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")

SCL_CLOUD = (8, 9, 10)
SCL_SHADOW = (3,)
SCL_NODATA = (0, 1)


def search(cfg, lon, lat):
    s2 = cfg["sentinel2"]
    feats, body = [], {"collections": [s2["collection"]], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                       "datetime": s2["date_range"], "limit": 100,
                       "query": {"eo:cloud_cover": {"lt": 90}}}
    url = s2["stac_url"]
    while url:
        r = requests.post(url, json=body, timeout=60)
        r.raise_for_status()
        page = r.json()
        feats += page["features"]
        nxt = [l for l in page.get("links", []) if l["rel"] == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return feats


def chip_bounds(item, lon, lat, size_m):
    epsg = item["properties"]["proj:epsg"]
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    x, y = xs[0], ys[0]
    x, y = round(x / 10) * 10, round(y / 10) * 10
    h = size_m / 2
    return epsg, (x - h, y - h, x + h, y + h)


def read_window(href, bounds, out_px, resampling=Resampling.nearest):
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read(1, window=win, out_shape=(out_px, out_px), resampling=resampling, boundless=True, fill_value=0)
        return arr, src.crs


def scale_offset(item, key):
    rb = (item["assets"][key].get("raster:bands") or [{}])[0]
    offset = float(rb.get("offset", 0.0))
    if item["properties"].get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return float(rb.get("scale", 1e-4)), offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", help="only this AOI")
    args = ap.parse_args()
    cfg = load_config()
    s2 = cfg["sentinel2"]
    px, size_m = s2["chip_size_px"], s2["chip_size_px"] * 10
    out_root = cfg["paths"]["raw"] / "sentinel2"

    for name, aoi in s2["aois"].items():
        if args.aoi and name != args.aoi:
            continue
        out_dir = out_root / name
        if (out_dir / "meta.json").exists():
            print(f"[{name}] already downloaded, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        items = search(cfg, aoi["lon"], aoi["lat"])
        by_date = {}
        for it in sorted(items, key=lambda f: f["properties"]["eo:cloud_cover"]):
            by_date.setdefault(it["properties"]["datetime"][:10], it)
        print(f"[{name}] {len(items)} passes, {len(by_date)} unique dates — scoring chip cloud from SCL ...")

        scored = []
        for date, it in sorted(by_date.items()):
            _, bounds = chip_bounds(it, aoi["lon"], aoi["lat"], size_m)
            try:
                scl, _ = read_window(it["assets"]["scl"]["href"], bounds, px // 4)
            except Exception as e:
                print(f"   {date}: SCL read failed ({e.__class__.__name__})")
                continue
            nodata = np.isin(scl, SCL_NODATA).mean()
            cloud = np.isin(scl, SCL_CLOUD).mean()
            shadow = np.isin(scl, SCL_SHADOW).mean()
            scored.append(dict(date=date, item=it, cloud=float(cloud), shadow=float(shadow), nodata=float(nodata)))

        valid = [s for s in scored if s["nodata"] < 0.01]
        clear = sorted([s for s in valid if s["cloud"] + s["shadow"] < 0.01], key=lambda s: s["cloud"] + s["shadow"])
        picked_clear, months = [], set()
        for s in clear:
            m = s["date"][:7]
            if m not in months:
                picked_clear.append(s), months.add(m)
            if len(picked_clear) == s2["clear_dates_per_aoi"]:
                break
        cloudy = sorted([s for s in valid if 0.15 <= s["cloud"] <= 0.65], key=lambda s: abs(s["cloud"] - 0.4))
        picked = [(s, "clear") for s in picked_clear] + [(s, "cloudy") for s in cloudy[: s2["cloudy_dates_per_aoi"]]]

        meta = {"aoi": name, **aoi, "files": []}
        for s, kind in picked:
            it = s["item"]
            epsg, bounds = chip_bounds(it, aoi["lon"], aoi["lat"], size_m)
            bands = []
            for key in ("green", "red", "nir"):
                arr, crs = read_window(it["assets"][key]["href"], bounds, px)
                bands.append(arr)
            scl, _ = read_window(it["assets"]["scl"]["href"], bounds, px)
            so = [scale_offset(it, k) for k in ("green", "red", "nir")]
            profile = dict(driver="GTiff", width=px, height=px, crs=crs,
                           transform=rasterio.transform.from_bounds(*bounds, px, px))
            fn = f"{s['date']}_{kind}.tif"
            write_raster(out_dir / fn, np.stack(bands + [scl]).astype(np.uint16), profile,
                         tags={"source": it["id"], "scale_offset": json.dumps(so)},
                         band_names=["green_B03", "red_B04", "nir_B08", "SCL"])
            meta["files"].append({"file": fn, "kind": kind, "date": s["date"], "item": it["id"], "epsg": epsg,
                                  "chip_cloud": s["cloud"], "chip_shadow": s["shadow"], "scale_offset": so})
            print(f"   saved {fn}  cloud={s['cloud']:.1%} shadow={s['shadow']:.1%}")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    total = sum(f.stat().st_size for f in out_root.rglob("*.tif")) / 1e6
    print(f"Done. Sentinel-2 chips on disk: {total:.0f} MB")


if __name__ == "__main__":
    main()
