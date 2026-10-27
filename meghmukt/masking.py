from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

CLEAR, CLOUD, SHADOW = 0, 1, 2


@dataclass
class MaskParams:
    bright: float = 0.26
    soft_width: float = 0.04
    flatness: float = 0.30
    ndvi_max: float = 0.25
    shadow_ratio: float = 0.65
    shadow_nir_min: float = 0.03
    shadow_reach: int = 60
    min_blob: int = 30
    buffer: int = 3

    def to_dict(self):
        return asdict(self)


def indices(x: np.ndarray):
    g, r, n = x
    eps = 1e-6
    bright = x.mean(0)
    flat = (x.max(0) - x.min(0)) / (bright + eps)
    ndvi = (n - r) / (n + r + eps)
    ndwi = (g - n) / (g + n + eps)
    return bright, flat, ndvi, ndwi


def cloud_probability(x: np.ndarray, p: MaskParams = MaskParams()) -> np.ndarray:
    bright, flat, ndvi, _ = indices(x)
    pb = np.clip((bright - (p.bright - p.soft_width)) / (2 * p.soft_width), 0, 1)
    pf = np.clip((p.flatness * 1.5 - flat) / (p.flatness), 0, 1)
    pn = np.clip((p.ndvi_max + 0.1 - ndvi) / 0.2, 0, 1)
    return (pb * pf * pn).astype(np.float32)


def _clean(m: np.ndarray, min_blob: int) -> np.ndarray:
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
    return keep[lab]


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(m)
    h, w = m.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out


def shadow_zone(cloud: np.ndarray, bright: np.ndarray, ref: float, reach: int, scale: int = 4):
    if cloud.mean() < 0.002 or cloud.mean() > 0.9:
        return None
    h, w = cloud.shape
    cs = cv2.resize(cloud.astype(np.uint8), (w // scale, h // scale), interpolation=cv2.INTER_AREA) > 0
    bs = cv2.resize(bright, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    best, best_val, vals = None, np.inf, []
    for ang in np.deg2rad(np.arange(0, 360, 10)):
        for d in range(3, max(reach // scale, 4), 2):
            dy, dx = int(round(d * np.sin(ang))), int(round(d * np.cos(ang)))
            fp = _shift(cs, dy, dx) & ~cs
            if fp.sum() < 20:
                continue
            v = bs[fp].mean() / ref
            vals.append(v)
            if v < best_val:
                best, best_val = (ang, d), v
    if best is None or best_val > 0.97 or np.median(vals) - best_val < 0.06:
        return None
    ang, d = best
    zone = np.zeros_like(cloud)
    for dd in np.linspace(0.5 * d, 1.5 * d, 9) * scale:
        zone |= _shift(cloud, int(round(dd * np.sin(ang))), int(round(dd * np.cos(ang))))
    zone = cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return zone & ~cloud


def detect(x: np.ndarray, p: MaskParams = MaskParams()) -> tuple[np.ndarray, np.ndarray]:
    prob = cloud_probability(x, p)
    cloud = _clean(prob > 0.5, p.min_blob)
    if p.buffer:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.buffer + 1,) * 2)
        cloud = cv2.dilate(cloud.astype(np.uint8), k).astype(bool)

    bright = x.mean(0)
    clear_ground = bright[~cloud]
    ref = np.median(clear_ground) if clear_ground.size > 100 else np.median(bright)
    dark = (bright < p.shadow_ratio * ref) & (x[2] > p.shadow_nir_min) & ~cloud
    zone = shadow_zone(cloud, bright, ref, p.shadow_reach)
    if zone is None:
        zone = cv2.distanceTransform((~cloud).astype(np.uint8), cv2.DIST_L2, 3) <= p.shadow_reach
    shadow = dark & zone
    shadow = _clean(shadow, p.min_blob)
    if p.buffer and shadow.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.buffer + 1,) * 2)
        shadow = cv2.dilate(shadow.astype(np.uint8), k).astype(bool) & ~cloud

    mask = np.zeros(x.shape[1:], np.uint8)
    mask[shadow] = SHADOW
    mask[cloud] = CLOUD
    return mask, prob


def mask_channels(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask == CLOUD, mask == SHADOW]).astype(np.float32)


def scl_to_mask(scl: np.ndarray) -> np.ndarray:
    m = np.zeros(scl.shape, np.uint8)
    m[np.isin(scl, (3,))] = SHADOW
    m[np.isin(scl, (8, 9, 10))] = CLOUD
    return m


def coverage(mask: np.ndarray) -> dict:
    return {"cloud": float((mask == CLOUD).mean()), "shadow": float((mask == SHADOW).mean())}


def detect_deep(x: np.ndarray, pixel_m: float = 10.0, buffer: int = 3) -> tuple[np.ndarray, np.ndarray]:
    from omnicloudmask import predict_from_array

    _, h, w = x.shape
    f = max(1, int(np.ceil(10.0 / pixel_m - 1e-6)))
    small = x if f == 1 else np.stack([cv2.resize(b, (w // f, h // f), interpolation=cv2.INTER_AREA) for b in x])
    rgn = np.stack([small[1], small[0], small[2]]).astype(np.float32) + 1e-4
    probs = predict_from_array(rgn, patch_size=min(1000, min(rgn.shape[1:]) // 32 * 32), patch_overlap=0,
                               export_confidence=True, softmax_output=True)
    cls = probs.argmax(0).astype(np.uint8)
    cloud_p = (probs[1] + probs[2]).astype(np.float32)
    if f > 1:
        cls = cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)
        cloud_p = cv2.resize(cloud_p, (w, h), interpolation=cv2.INTER_LINEAR)
    cloud, shadow = np.isin(cls, (1, 2)), cls == 3
    if buffer:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer + 1,) * 2)
        cloud = cv2.dilate(cloud.astype(np.uint8), k).astype(bool)
        shadow = cv2.dilate(shadow.astype(np.uint8), k).astype(bool) & ~cloud
    mask = np.zeros((h, w), np.uint8)
    mask[shadow] = SHADOW
    mask[cloud] = CLOUD
    return mask, cloud_p


def detect_auto(x: np.ndarray, method: str = "deep", pixel_m: float = 10.0, params: MaskParams | None = None):
    if method == "deep":
        try:
            return detect_deep(x, pixel_m)
        except ImportError:
            pass
    return detect(x, params or MaskParams())


def pixel_size(profile: dict | None, default: float = 5.0) -> float:
    try:
        return float(abs(profile["transform"].a))
    except (TypeError, KeyError, AttributeError):
        return default
