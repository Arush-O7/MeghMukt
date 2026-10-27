from __future__ import annotations

import cv2
import numpy as np
from scipy.special import gamma as G

PATCH = 32
_ALPHAS = np.arange(0.2, 10.0, 0.001)
_R_GGD = G(1 / _ALPHAS) * G(3 / _ALPHAS) / G(2 / _ALPHAS) ** 2
_R_AGGD = G(2 / _ALPHAS) ** 2 / (G(1 / _ALPHAS) * G(3 / _ALPHAS))


def sharpness_ratio(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    def grad(x):
        g = x.mean(0)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    if region.sum() < 50:
        return float("nan")
    return float(grad(pred)[region].mean() / (grad(gt)[region].mean() + 1e-9))


def _to_gray255(x):
    return np.clip(x.mean(0) / 0.5, 0, 1).astype(np.float32) * 255


def _mscn(img):
    mu = cv2.GaussianBlur(img, (7, 7), 7 / 6)
    sd = np.sqrt(np.abs(cv2.GaussianBlur(img * img, (7, 7), 7 / 6) - mu * mu))
    return (img - mu) / (sd + 1.0)


def _ggd(x):
    s2 = np.mean(x * x)
    r = s2 / (np.mean(np.abs(x)) ** 2 + 1e-12)
    return _ALPHAS[np.argmin((_R_GGD - r) ** 2)], s2


def _aggd(x):
    l, r_ = x[x < 0], x[x >= 0]
    sl = np.sqrt(np.mean(l * l)) if l.size else 1e-6
    sr = np.sqrt(np.mean(r_ * r_)) if r_.size else 1e-6
    gh = sl / sr
    rh = np.mean(np.abs(x)) ** 2 / (np.mean(x * x) + 1e-12)
    rn = rh * (gh ** 3 + 1) * (gh + 1) / (gh ** 2 + 1) ** 2
    a = _ALPHAS[np.argmin((_R_AGGD - rn) ** 2)]
    mean = (sr - sl) * G(2 / a) / G(1 / a) * np.sqrt(G(1 / a) / G(3 / a))
    return [a, mean, sl * sl, sr * sr]


def _patch_feats(m):
    f = list(_ggd(m))
    for sh in ((0, 1), (1, 0), (1, 1), (1, -1)):
        f += _aggd(m * np.roll(m, sh, axis=(0, 1)))
    return f


def patch_features(x: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    img = _to_gray255(x)
    feats = []
    for scale in (1, 2):
        im = img if scale == 1 else cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        m = _mscn(im)
        p = PATCH // scale
        rows = []
        for y in range(0, m.shape[0] - p + 1, p):
            for x0 in range(0, m.shape[1] - p + 1, p):
                rows.append(_patch_feats(m[y:y + p, x0:x0 + p]))
        feats.append(np.array(rows))
    F = np.concatenate(feats, 1)
    if keep is not None:
        H, W = keep.shape
        frac = [keep[y:y + PATCH, x0:x0 + PATCH].mean() for y in range(0, H - PATCH + 1, PATCH)
                for x0 in range(0, W - PATCH + 1, PATCH)]
        F = F[np.array(frac) > 0.5]
    return F


class NIQE:
    def __init__(self, mu, cov):
        self.mu, self.cov = np.asarray(mu), np.asarray(cov)

    @classmethod
    def fit(cls, pristine_images, min_sharp=0.75):
        F = []
        for x in pristine_images:
            img = _to_gray255(x)
            sd = np.sqrt(np.abs(cv2.GaussianBlur(img * img, (7, 7), 7 / 6) - cv2.GaussianBlur(img, (7, 7), 7 / 6) ** 2))
            f = patch_features(x)
            sharp = [sd[y:y + PATCH, x0:x0 + PATCH].mean() for y in range(0, img.shape[0] - PATCH + 1, PATCH)
                     for x0 in range(0, img.shape[1] - PATCH + 1, PATCH)]
            sharp = np.array(sharp)
            F.append(f[sharp >= min_sharp * sharp.max()])
        F = np.concatenate(F)
        return cls(F.mean(0), np.cov(F, rowvar=False))

    def score(self, F: np.ndarray) -> float:
        if len(F) < 10:
            return float("nan")
        d = F.mean(0) - self.mu
        S = (self.cov + np.cov(F, rowvar=False)) / 2
        return float(np.sqrt(d @ np.linalg.pinv(S) @ d))

    def save(self, path):
        np.savez(path, mu=self.mu, cov=self.cov)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mu"], d["cov"])
