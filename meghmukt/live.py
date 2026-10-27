from __future__ import annotations

import datetime as dt

import numpy as np
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

from .reference import STAC, _read_on_grid, fetch_sentinel2_reference

NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "MeghMukt-student-project/1.0 (cloud removal demo)"}

PLACES = {
    "Guwahati": (26.15, 91.70), "Majuli": (26.95, 94.20), "Kaziranga": (26.62, 93.35), "Tezpur": (26.63, 92.80),
    "Dibrugarh": (27.48, 94.91), "Jorhat": (26.75, 94.21), "Dhubri": (26.02, 89.98), "Silchar": (24.82, 92.80),
    "Shillong": (25.57, 91.88), "Tura": (25.51, 90.22), "Itanagar": (27.08, 93.61), "Pasighat": (28.07, 95.33),
    "Kohima": (25.67, 94.11), "Dimapur": (25.90, 93.73), "Imphal": (24.80, 93.94), "Aizawl": (23.73, 92.72),
    "Agartala": (23.83, 91.28), "Gangtok": (27.33, 88.61),
}


def geocode(name: str):
    r = requests.get(NOMINATIM, params={"q": name, "format": "json", "limit": 1, "countrycodes": "in"},
                     headers=HEADERS, timeout=20)
    js = r.json()
    if not js:
        return None
    return float(js[0]["lat"]), float(js[0]["lon"]), js[0]["display_name"]


def search_cloudy(lat, lon, start: str, end: str, cc_min=10, cc_max=70, limit=40):
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit,
            "query": {"eo:cloud_cover": {"gte": cc_min, "lte": cc_max}},
            "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    feats = requests.post(STAC, json=body, timeout=60).json().get("features", [])
    return [dict(id=f["id"], date=f["properties"]["datetime"][:10], cloud=float(f["properties"]["eo:cloud_cover"]), item=f)
            for f in feats]


def _epsg(item, lon):
    p = item["properties"]
    if p.get("proj:epsg"):
        return int(p["proj:epsg"])
    if p.get("proj:code"):
        return int(str(p["proj:code"]).split(":")[-1])
    return 32600 + int((lon + 180) // 6) + 1


def chip_profile(item, lat, lon, size=512, res=10.0):
    epsg = _epsg(item, lon)
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    x0 = round((xs[0] - size * res / 2) / res) * res
    y0 = round((ys[0] + size * res / 2) / res) * res
    return dict(driver="GTiff", crs=CRS.from_epsg(epsg), transform=from_origin(x0, y0, res, res),
                width=size, height=size, count=3, dtype="float32")


def read_chip(item, prof):
    refl = []
    for key in ("green", "red", "nir"):
        dn = _read_on_grid(item["assets"][key]["href"], prof, Resampling.bilinear).astype(np.float32)
        refl.append(dn * 1e-4 if item["properties"].get("earthsearch:boa_offset_applied", True) else dn * 1e-4 - 0.1)
    arr = np.clip(np.stack(refl), 0, None)
    return arr, float((arr <= 0).all(0).mean())


def reference_and_check(prof, harm, date: str, before_days=240, after_days=20, check=True):
    d = dt.date.fromisoformat(date)
    ref, ref_info = fetch_sentinel2_reference(
        prof, harm, f"{d - dt.timedelta(days=before_days)}/{d - dt.timedelta(days=1)}", log=lambda *a: None, order="latest")
    chk, chk_info = (None, {"reason": "not requested"})
    if check:
        chk, chk_info = fetch_sentinel2_reference(
            prof, harm, f"{d + dt.timedelta(days=1)}/{d + dt.timedelta(days=after_days)}", max_candidates=4,
            log=lambda *a: None, order="earliest")
    return ref, ref_info, chk, chk_info


def chip_cloud(item, prof, factor=8):
    import rasterio
    small = dict(prof, width=prof["width"] // factor, height=prof["height"] // factor,
                 transform=prof["transform"] * rasterio.Affine.scale(factor))
    scl = _read_on_grid(item["assets"]["scl"]["href"], small, Resampling.nearest)
    return float(np.isin(scl, (8, 9, 10)).mean()), float((scl == 0).mean())


def _refl(item, key, dn):
    return dn * 1e-4 if item["properties"].get("earthsearch:boa_offset_applied", True) else dn * 1e-4 - 0.1


def read_mosaic(items, key, prof, resampling=Resampling.bilinear):
    out = None
    for it in items:
        a = _read_on_grid(it["assets"][key]["href"], prof, resampling).astype(np.float32)
        if key != "scl":
            a = np.where(a > 0, np.clip(_refl(it, key, a), 1e-4, None), 0)
        out = a if out is None else np.where(out > 0, out, a)
    return out


def find_clear_date(lat, lon, window_m, years=2, max_dates=8):
    end = dt.date.today()
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{end - dt.timedelta(days=365 * years)}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"eo:cloud_cover": {"lt": 15}}, "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    by_date = {}
    for f in requests.post(STAC, json=body, timeout=60).json().get("features", []):
        by_date.setdefault(f["properties"]["datetime"][:10], []).append(f)
    best = None
    for date, items in list(by_date.items())[:max_dates]:
        prof = chip_profile(items[0], lat, lon, size=64, res=window_m / 64)
        try:
            scl = read_mosaic(items, "scl", prof, Resampling.nearest)
        except Exception:
            continue
        bad = float(np.isin(scl, (0, 3, 8, 9, 10)).mean())
        if bad < 0.02:
            return items
        if best is None or bad < best[0]:
            best = (bad, items)
    return best[1] if best is not None and best[0] < 0.1 else None


def _stretch8(x):
    lo, hi = np.percentile(x, [2, 98])
    return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def locate_picture(rgb8: np.ndarray, layout: str, lat: float, lon: float, window_km: float = 20.0,
                   cloud_mask: np.ndarray | None = None):
    import cv2
    from rasterio import Affine
    items = find_clear_date(lat, lon, window_km * 1000)
    if items is None:
        return None, "no cloud-free Sentinel-2 pass of this area in the last two years"
    item, n = items[0], 1200
    win = chip_profile(item, lat, lon, size=n, res=window_km * 1000 / n)
    bands = {k: read_mosaic(items, k, win) for k in ("green", "red", "nir")}
    if layout == "fcc":
        view = np.stack([_stretch8(bands["nir"]), _stretch8(bands["red"]), _stretch8(bands["green"])], -1)
    else:
        view = np.stack([_stretch8(bands["red"]), _stretch8(bands["green"]), _stretch8(bands["green"])], -1)
        pic_view = np.stack([rgb8[..., 0], rgb8[..., 1], rgb8[..., 1]], -1)
    if layout == "fcc":
        pic_view = rgb8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_ref = clahe.apply(cv2.cvtColor(view, cv2.COLOR_RGB2GRAY))
    g_pic = clahe.apply(cv2.cvtColor(np.ascontiguousarray(pic_view), cv2.COLOR_RGB2GRAY))
    keep = None if cloud_mask is None else ((cloud_mask == 0) * 255).astype(np.uint8)
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, d1 = sift.detectAndCompute(g_pic, keep)
    kp2, d2 = sift.detectAndCompute(g_ref, None)
    if d1 is None or d2 is None or len(kp1) < 10:
        return None, "too few clear features in the picture"
    good = [m for m, k in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if m.distance < 0.75 * k.distance]
    if len(good) < 10:
        return None, f"the picture was not found in the {window_km:g} km area (only {len(good)} candidate matches)"
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    n_in = int(inl.sum()) if inl is not None else 0
    if M is None or n_in < 12:
        return None, f"the picture was not found in the {window_km:g} km area ({n_in} consistent matches)"
    h, w = rgb8.shape[:2]
    transform = win["transform"] * Affine(*M[0], *M[1])
    pic_prof = dict(driver="GTiff", crs=win["crs"], transform=transform, width=w, height=h, count=3, dtype="float32")
    refl = np.stack([read_mosaic(items, k, pic_prof) for k in ("green", "red", "nir")])
    px = float(np.sqrt(abs(transform.a * transform.e - transform.b * transform.d)))
    return dict(profile=pic_prof, refl=refl, item=item["id"], date=item["properties"]["datetime"][:10],
                inliers=n_in, matches=len(good), pixel_m=px, nodata=float((refl <= 0).all(0).mean())), "ok"
