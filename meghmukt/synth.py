from __future__ import annotations

import cv2
import numpy as np

from .masking import CLEAR, CLOUD, SHADOW

DEFAULT_CLOUD_SPECTRA = np.array([[0.45, 0.43, 0.46], [0.52, 0.51, 0.53], [0.40, 0.38, 0.42],
                                  [0.55, 0.55, 0.57], [0.48, 0.46, 0.50]], np.float32)


def fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 6,
                  persistence: float = 0.55, start: int = 3) -> np.ndarray:
    total = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        res = start * 2 ** o
        grid = rng.random((res + 1, res + 1)).astype(np.float32)
        total += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= persistence
    total -= total.min()
    return total / (total.max() + 1e-6)


class CloudSynthesiser:
    def __init__(self, cloud_spectra: np.ndarray | None = None, coverage=(0.15, 0.70),
                 shadow_prob: float = 0.7):
        self.spectra = DEFAULT_CLOUD_SPECTRA if cloud_spectra is None else np.asarray(cloud_spectra, np.float32)
        self.coverage = coverage
        self.shadow_prob = shadow_prob

    def opacity(self, h, w, rng, cover):
        n = fractal_noise(h, w, rng)
        t = np.quantile(n, 1 - cover)
        soft = rng.uniform(0.04, 0.15)
        a = np.clip((n - t) / soft + 0.5, 0, 1) ** rng.uniform(1.0, 1.6)
        if rng.random() < 0.3:
            a *= rng.uniform(0.35, 0.75)
        if rng.random() < 0.3:
            a = np.maximum(a, rng.uniform(0.05, 0.2) * fractal_noise(h, w, rng, start=2))
        return a.astype(np.float32)

    def __call__(self, ground: np.ndarray, rng: np.random.Generator):
        _, h, w = ground.shape
        cover = rng.uniform(*self.coverage)
        a = self.opacity(h, w, rng, cover)

        spec = self.spectra[rng.integers(len(self.spectra))] * rng.uniform(0.9, 1.1)
        tex = 0.88 + 0.24 * fractal_noise(h, w, rng, start=4)
        cloud = spec[:, None, None] * tex[None]

        shadow_a = np.zeros((h, w), np.float32)
        if rng.random() < self.shadow_prob:
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.06, 0.25) * h
            dx, dy = int(dist * np.cos(ang)), int(dist * np.sin(ang))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shadow_a = cv2.warpAffine(np.clip(a * 1.3, 0, 1), M, (w, h), borderMode=cv2.BORDER_CONSTANT)
            shadow_a = cv2.GaussianBlur(shadow_a, (0, 0), 2.0)
        k = rng.uniform(0.3, 0.6)
        shadowed = ground * (1 - k * shadow_a)[None]

        cloudy = (1 - a)[None] * shadowed + a[None] * cloud
        cloudy += rng.normal(0, 0.002, cloudy.shape).astype(np.float32)

        mask = np.full((h, w), CLEAR, np.uint8)
        mask[(k * shadow_a > 0.08) & (a < 0.15)] = SHADOW
        mask[a >= 0.15] = CLOUD
        return np.clip(cloudy, 0, 1).astype(np.float32), mask, a
