from __future__ import annotations

import os

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .io import Harmoniser, liss4_dn_to_norm, read_raster, reproject_to

STAC = "https://earth-search.aws.element84.com/v1/search"
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


def to_liss4_scale(arr: np.ndarray, sensor: str, harm: Harmoniser | None) -> np.ndarray:
    if sensor == "liss4":
        return arr.astype(np.float32) if np.nanmax(arr) <= 1.5 else liss4_dn_to_norm(arr)
    refl = arr.astype(np.float32) * (1e-4 if np.nanmax(arr) > 1.5 else 1.0)
    return harm(refl) if harm is not None else refl


def load_reference(path, sensor, target_profile, harm, bands=(0, 1, 2)):
    arr, prof = read_raster(path)
    arr = arr[list(bands)]
    if (prof["crs"], prof["transform"], prof["width"], prof["height"]) != (
            target_profile["crs"], target_profile["transform"], target_profile["width"], target_profile["height"]):
        arr = reproject_to(arr, prof, target_profile)
    return to_liss4_scale(np.nan_to_num(arr), sensor, harm)


def _read_on_grid(href, prof, resampling):
    with rasterio.open(href) as src, WarpedVRT(src, crs=prof["crs"], transform=prof["transform"],
                                               width=prof["width"], height=prof["height"],
                                               resampling=resampling) as vrt:
        return vrt.read(1)


def fetch_sentinel2_reference(target_profile, harm: Harmoniser, date_range="2023-10-01/2025-03-31",
                              max_candidates=12, log=print, order="cloud"):
    w, s, e, n = transform_bounds(target_profile["crs"], "EPSG:4326", *_bounds(target_profile))
    a, b = date_range.split("/")
    body = {"collections": ["sentinel-2-l2a"], "bbox": [w, s, e, n], "datetime": f"{a}T00:00:00Z/{b}T23:59:59Z",
            "limit": 100, "query": {"eo:cloud_cover": {"lt": 20}}}
    feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
    if order == "cloud":
        feats.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
    else:
        feats.sort(key=lambda f: f["properties"]["datetime"], reverse=order == "latest")
    small = dict(target_profile, width=max(target_profile["width"] // 8, 16), height=max(target_profile["height"] // 8, 16),
                 transform=target_profile["transform"] * rasterio.Affine.scale(
                     target_profile["width"] / max(target_profile["width"] // 8, 16),
                     target_profile["height"] / max(target_profile["height"] // 8, 16)))
    for f in feats[:max_candidates]:
        try:
            scl = _read_on_grid(f["assets"]["scl"]["href"], small, Resampling.nearest)
        except Exception:
            continue
        bad = np.isin(scl, (0, 1, 3, 8, 9, 10)).mean()
        log(f"  candidate {f['id']}: {bad:.1%} cloud/shadow/no-data in footprint")
        if bad < 0.02:
            refl = []
            for key in ("green", "red", "nir"):
                dn = _read_on_grid(f["assets"][key]["href"], target_profile, Resampling.bilinear).astype(np.float32)
                refl.append(dn * 1e-4 if f["properties"].get("earthsearch:boa_offset_applied", True) else dn * 1e-4 - 0.1)
            return harm(np.stack(refl)), {"reference_item": f["id"], "reference_date": f["properties"]["datetime"][:10], "reference_bad_fraction": float(bad)}
    return None, {"reason": "no clear Sentinel-2 pass found over this footprint"}


def _bounds(prof):
    t, w, h = prof["transform"], prof["width"], prof["height"]
    xs = [t.c, t.c + t.a * w]
    ys = [t.f, t.f + t.e * h]
    return min(xs), min(ys), max(xs), max(ys)
