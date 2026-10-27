from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

DARK_QUANTILE = 0.7


def dark_levels(band: np.ndarray, brightness: np.ndarray) -> list[float]:
    sel = brightness.ravel() <= np.quantile(brightness, DARK_QUANTILE)
    lo, med = np.percentile(band.ravel()[sel], [5, 50])
    return [float(lo), float(med)]


def _feats(b, g, r):
    e = 1e-6
    return np.stack([b, g, r, (g - r) / (g + r + e), (g - b) / (g + b + e), (r - b) / (r + b + e)])


class NirFromRGB:

    def __init__(self, params: dict):
        self.params = params

    @classmethod
    def fit(cls, b, g, r, nir):
        from sklearn.neural_network import MLPRegressor
        F = _feats(b, g, r)
        mu, sd = F.mean(1), F.std(1) + 1e-6
        m = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=300, early_stopping=True, random_state=0)
        m.fit((F.T - mu) / sd, nir)
        return cls({"mu": mu.tolist(), "sd": sd.tolist(),
                    "W": [w.tolist() for w in m.coefs_], "b": [c.tolist() for c in m.intercepts_]})

    def predict(self, b, g, r):
        shape = np.shape(g)
        h = (_feats(*(np.ravel(x) for x in (b, g, r))).T - np.asarray(self.params["mu"])) / np.asarray(self.params["sd"])
        for i, (W, c) in enumerate(zip(self.params["W"], self.params["b"])):
            h = h @ np.asarray(W) + np.asarray(c)
            if i < len(self.params["W"]) - 1:
                h = np.maximum(h, 0)
        return np.clip(h.ravel(), 0, None).reshape(shape).astype(np.float32)


def visible_cloud_mask(rgb8: np.ndarray, buffer: int = 6) -> np.ndarray:
    import cv2
    x = _srgb_to_linear(rgb8.astype(np.float32) / 255.0)
    bright = x.mean(-1)
    white = (x.max(-1) - x.min(-1)) / (bright + 1e-6)
    m = (bright > max(2.5 * np.median(bright), 0.25)) & (white < 0.35)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if buffer:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer + 1, 2 * buffer + 1)))
    return m.astype(np.uint8)


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def load_web_image(data: bytes, layout: str, harm, calib: dict, max_side: int = 1024):
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    rgb8 = np.asarray(im)
    pic = _srgb_to_linear(rgb8.astype(np.float32) / 255.0).transpose(2, 0, 1)
    names = ("nir", "red", "green") if layout == "fcc" else ("red", "green", "blue")
    bright = pic.mean(0)
    cloud = visible_cloud_mask(rgb8, buffer=0) > 0
    two_anchor = cloud.mean() > 0.005 and "cloud_levels" in calib
    land = bright <= np.quantile(bright[~cloud] if two_anchor else bright, DARK_QUANTILE)
    refl, maps = {}, {}
    for name, band in zip(names, pic):
        if two_anchor:
            a, b = float(np.median(band[land & ~cloud])), float(np.median(band[cloud]))
            ta, tb = calib["levels"][name][1], calib["cloud_levels"][name]
        else:
            a, b = dark_levels(band, bright)
            ta, tb = calib["levels"][name]
        maps[name] = (a, max(b, a + 1e-4), ta, tb)
        refl[name] = np.clip(ta + (band - a) * (tb - ta) / max(b - a, 1e-4), 0, 1.5).astype(np.float32)
    if layout != "fcc":
        nir = NirFromRGB(calib["nir_model"]).predict(refl["blue"], refl["green"], refl["red"])
        if "cloud_levels" in calib:
            vis = (refl["blue"] + refl["green"] + refl["red"]) / 3
            cl, gl = calib["cloud_levels"], calib["levels"]
            c_vis = (cl["blue"] + cl["green"] + cl["red"]) / 3
            g_vis = (gl["blue"][1] + gl["green"][1] + gl["red"][1]) / 3
            w = np.clip((vis - g_vis) / (c_vis - g_vis), 0, 1)
            nir = (1 - w) * nir + w * vis * cl["nir"] / c_vis
        refl["nir"] = nir.astype(np.float32)
    info = {"layout": layout, "names": names, "maps": maps}
    return harm(np.stack([refl["green"], refl["red"], refl["nir"]])).astype(np.float32), rgb8, info


def to_picture(out: np.ndarray, mask: np.ndarray, rgb8: np.ndarray, info: dict, harm) -> np.ndarray:
    refl = harm.inverse(out)
    bands = {"green": refl[0], "red": refl[1], "nir": refl[2]}
    lin = []
    for name in info["names"]:
        if name == "blue":
            lin.append(None)
            continue
        lo, med, tlo, tmed = info["maps"][name]
        lin.append(lo + (bands[name] - tlo) * (med - lo) / (tmed - tlo))
    if info["layout"] != "fcc":
        orig = _srgb_to_linear(rgb8.astype(np.float32) / 255.0)
        clear = mask == 0
        A = np.stack([orig[..., 0][clear], orig[..., 1][clear], np.ones(clear.sum())], 1)
        w, *_ = np.linalg.lstsq(A, orig[..., 2][clear], rcond=None)
        lin[2] = w[0] * lin[0] + w[1] * lin[1] + w[2]
    rec = (_linear_to_srgb(np.stack(lin, -1)) * 255).round().astype(np.uint8)
    return np.where((mask > 0)[..., None], rec, rgb8)
