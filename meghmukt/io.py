from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
LISS4_DN_MAX = 1023.0


def load_config(path: str | Path | None = None) -> dict:
    path = path or os.environ.get("MEGHMUKT_CONFIG") or ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = (ROOT / v).resolve()
    return cfg


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.profile.copy()


def write_raster(path: str | Path, arr: np.ndarray, profile: dict, tags: dict | None = None,
                 band_names: list[str] | None = None) -> None:
    arr = np.asarray(arr)
    prof = profile.copy()
    prof.update(count=arr.shape[0], dtype=arr.dtype.name, compress="deflate", nodata=None)
    prof.pop("blockxsize", None), prof.pop("blockysize", None)
    prof.update(tiled=True, blockxsize=256, blockysize=256)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
        for i, name in enumerate(band_names or []):
            dst.set_band_description(i + 1, name)


def reproject_to(src_arr: np.ndarray, src_profile: dict, dst_profile: dict,
                 resampling=Resampling.bilinear) -> np.ndarray:
    out = np.zeros((src_arr.shape[0], dst_profile["height"], dst_profile["width"]), np.float32)
    reproject(src_arr, out, src_transform=src_profile["transform"], src_crs=src_profile["crs"],
              dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
              resampling=resampling, src_nodata=None, dst_nodata=np.nan)
    return out


def liss4_dn_to_norm(dn: np.ndarray) -> np.ndarray:
    return np.clip(dn.astype(np.float32) / LISS4_DN_MAX, 0, 1)


def norm_to_liss4_dn(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * LISS4_DN_MAX).astype(np.uint16)


def s2_dn_to_reflectance(dn: np.ndarray, scale: float = 1e-4, offset: float = 0.0) -> np.ndarray:
    return dn.astype(np.float32) * scale + offset


class Harmoniser:

    def __init__(self, gain=(1.0, 1.0, 1.0), bias=(0.0, 0.0, 0.0)):
        self.gain = np.asarray(gain, np.float32)[:, None, None]
        self.bias = np.asarray(bias, np.float32)[:, None, None]

    @classmethod
    def fit_anchors(cls, s2_ground, s2_cloud, liss_ground, liss_cloud) -> "Harmoniser":
        sg, sc = np.median(s2_ground, 1), np.median(s2_cloud, 1)
        lg, lc = np.median(liss_ground, 1), np.median(liss_cloud, 1)
        gain = (lc - lg) / (sc - sg + 1e-6)
        return cls(gain, lg - gain * sg)

    def __call__(self, refl: np.ndarray) -> np.ndarray:
        return np.clip(self.gain * refl + self.bias, 0, 1).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.bias) / self.gain, 0, 1.5).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"gain": self.gain.ravel().tolist(),
                                          "bias": self.bias.ravel().tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["gain"], d["bias"])


_BLUE = {}


def _blue_tools():
    if "v" not in _BLUE:
        proc = ROOT / "data" / "processed"
        try:
            from .bluesynth import BlueSynth
            _BLUE["v"] = (BlueSynth.load(proc / "blue_fit.json"), Harmoniser.load(proc / "harmoniser.json"))
        except (FileNotFoundError, ImportError):
            _BLUE["v"] = None
    return _BLUE["v"]


def natural_colour(x: np.ndarray) -> np.ndarray:
    tools = _blue_tools()
    if tools is None:
        g, r, n = x
        return np.stack([r, g, np.clip(0.85 * g + 0.25 * r - 0.12 * n, 0, 1)])
    blue_model, harm = tools
    refl = harm.inverse(x)
    return np.stack([refl[1], refl[0], blue_model.predict(refl)])


def _compose(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "natural":
        return natural_colour(x)
    g, r, n = x
    return np.stack([n, r, g])


def display_limits(x: np.ndarray, mode: str = "fcc", p=(2, 98)):
    flat = _compose(x, mode).reshape(3, -1)
    flat = flat[:, np.isfinite(flat).all(0)]
    return np.percentile(flat, p[0], axis=1), np.percentile(flat, p[1], axis=1)


def to_display(x: np.ndarray, mode: str = "fcc", limits=None) -> np.ndarray:
    rgb = _compose(x, mode)
    lo, hi = limits if limits is not None else display_limits(x, mode)
    out = np.clip((rgb - lo[:, None, None]) / (hi - lo + 1e-6)[:, None, None], 0, 1)
    return (np.nan_to_num(out).transpose(1, 2, 0) ** 0.9 * 255).astype(np.uint8)
