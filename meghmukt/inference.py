from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from .dataset import Normaliser, make_condition
from .io import norm_to_liss4_dn, write_raster
from .masking import CLEAR
from .models import ClassicalFill, ConditionalDiffusion

CONF_TAU = 0.015
DEPTH_SCALE = 10.0
DEPTH_ONLY_U = 0.01


def depth_factor(region: np.ndarray) -> np.ndarray:
    d = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    return (1 - np.exp(-d / DEPTH_SCALE)).astype(np.float32)


def uncertainty_map(region: np.ndarray, spread: np.ndarray | None) -> np.ndarray:
    depth = depth_factor(region)
    u = spread * depth if spread is not None else DEPTH_ONLY_U * depth
    return np.where(region, u, 0).astype(np.float32)


def confidence_from(u: np.ndarray, region: np.ndarray) -> np.ndarray:
    return np.where(region, np.exp(-u / CONF_TAU), 1.0).astype(np.float32)


def _dihedral(x, k, flip):
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def _undo(x, k, flip):
    x = torch.flip(x, dims=(-1,)) if flip else x
    return torch.rot90(x, -k, dims=(-2, -1))


@torch.no_grad()
def predict(model, cond: torch.Tensor, tta: bool = True, n_samples: int = 4, steps: int = 20, seed: int = 0):
    if isinstance(model, ClassicalFill):
        out = model(cond)
        return out, torch.zeros_like(out)
    outs = []
    if isinstance(model, ConditionalDiffusion):
        for s in range(n_samples):
            g = torch.Generator(device=cond.device).manual_seed(seed + s)
            k, flip = (s % 4, s >= 4) if tta else (0, False)
            outs.append(_undo(model.sample(_dihedral(cond, k, flip), steps, g), k, flip))
    else:
        views = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]
        for k, f in views:
            outs.append(_undo(model(_dihedral(cond, k, f)), k, f))
    stack = torch.stack(outs)
    return stack.mean(0), stack.std(0) if len(outs) > 1 else torch.zeros_like(outs[0])


def _window(size, overlap, floor=0.08):
    if overlap == 0:
        return np.ones((size, size), np.float32)
    ramp = np.ones(size, np.float32)
    r = np.linspace(floor, 1, overlap, dtype=np.float32)
    ramp[:overlap], ramp[-overlap:] = r, r[::-1]
    return np.outer(ramp, ramp)


@torch.no_grad()
def reconstruct_scene(model, cloudy: np.ndarray, mask: np.ndarray, ref: np.ndarray | None, norm: Normaliser,
                      device, tile=256, overlap=64, tta=True, n_samples=4, steps=20, batch=8, progress=None,
                      conf_model=None):
    cond_full = make_condition(cloudy, mask, ref, norm, ref is not None)
    _, H, W = cloudy.shape
    t = min(tile, H, W) // 8 * 8
    ov = min(overlap, t // 4)
    ys = list(range(0, H - t + 1, t - ov))
    xs = list(range(0, W - t + 1, t - ov))
    if ys[-1] + t < H:
        ys.append(H - t)
    if xs[-1] + t < W:
        xs.append(W - t)
    win = _window(t, ov)
    acc = np.zeros((3, H, W), np.float32)
    acc_s = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    std_scale = torch.as_tensor(norm.std, device=device)
    todo = [(y, x) for y in ys for x in xs if (mask[y:y + t, x:x + t] != CLEAR).any()]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        c = torch.from_numpy(np.stack([cond_full[:, y:y + t, x:x + t] for y, x in chunk])).to(device)
        mean, _ = predict(model, c, tta, n_samples, steps)
        mean = norm.inverse(mean).cpu().numpy()
        if conf_model is not None:
            _, spread = predict(conf_model, c, True, n_samples, steps)
            spread = (spread * std_scale).mean(1).cpu().numpy()
        for j, (y, x) in enumerate(chunk):
            acc[:, y:y + t, x:x + t] += mean[j] * win
            if conf_model is not None:
                acc_s[y:y + t, x:x + t] += spread[j] * win
            wsum[y:y + t, x:x + t] += win
        if progress:
            progress(min(1.0, (i + len(chunk)) / max(len(todo), 1)))
    covered = wsum > 0
    out = cloudy.copy()
    out[:, covered] = acc[:, covered] / wsum[covered]
    region = mask != CLEAR
    spread = None
    if conf_model is not None:
        spread = np.zeros((H, W), np.float32)
        spread[covered] = acc_s[covered] / wsum[covered]
    unc = uncertainty_map(region, spread)
    return np.clip(out, 0, 1), unc, confidence_from(unc, region)


def export_analysis_ready(path, out, mask, conf, profile, model_name, extra_tags=None):
    arr = np.concatenate([norm_to_liss4_dn(out), mask[None].astype(np.uint16),
                          np.round(conf[None] * 100).astype(np.uint16)])
    tags = {"PRODUCT": "MeghMukt cloud-free LISS-IV", "MODEL": model_name,
            "CREATED_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "BANDS": "1 Green, 2 Red, 3 NIR (DN, /1023 for normalised); 4 quality mask; 5 confidence %",
            "RECONSTRUCTED_FRACTION": f"{(mask > 0).mean():.4f}"}
    tags.update(extra_tags or {})
    prof = {k: v for k, v in profile.items() if k in ("driver", "crs", "transform", "width", "height")}
    prof.setdefault("driver", "GTiff")
    write_raster(path, arr, prof, tags=tags, band_names=["green", "red", "nir", "quality_mask", "confidence_pct"])
