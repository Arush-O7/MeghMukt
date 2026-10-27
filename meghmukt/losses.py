from __future__ import annotations

import torch
import torch.nn.functional as F


def _gauss(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, data_range=1.0):
    c = x.shape[1]
    w = _gauss(device=x.device).repeat(c, 1, 1, 1)
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def pixel_loss(pred, gt, m, charbonnier=False, eps=1e-3, valid=None):
    w = (1 + 4 * m) * (1 if valid is None else valid)
    d = pred - gt
    err = torch.sqrt(d * d + eps * eps) if charbonnier else d.abs()
    return (err * w).sum() / (w.sum() * pred.shape[1])


def ssim_loss(pred, gt, valid=None):
    smap = ssim_map(pred, gt, data_range=0.5).mean(1, keepdim=True)
    if valid is None:
        return 1 - smap.mean()
    return ((1 - smap) * valid).sum() / (valid.sum() + 1e-6)


def sam_loss(pred, gt, m, eps=1e-6):
    cos = (pred * gt).sum(1, keepdim=True) / (pred.norm(dim=1, keepdim=True) * gt.norm(dim=1, keepdim=True) + eps)
    return ((1 - cos) * m).sum() / (m.sum() + eps)


def ndvi(x, floor=0.01):
    x = x.clamp(min=floor)
    return (x[:, 2:3] - x[:, 1:2]) / (x[:, 2:3] + x[:, 1:2]).clamp(min=0.05)


def ndvi_loss(pred, gt, m, eps=1e-6):
    return ((ndvi(pred) - ndvi(gt)).abs() * m).sum() / (m.sum() + eps)


def gan_loss(logits, real: bool):
    return F.mse_loss(logits, torch.full_like(logits, 1.0 if real else 0.0))
