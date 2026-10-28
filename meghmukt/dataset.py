from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .masking import mask_channels
from .synth import CloudSynthesiser, fractal_noise


def _global_match(cloudy, clear, ref):
    out = ref.copy()
    for b in range(3):
        x, y = ref[b][clear], cloudy[b][clear]
        if x.std() < 1e-6:
            continue
        a = np.cov(x, y, bias=True)[0, 1] / x.var()
        c = y.mean() - a * x.mean()
        if 0.3 < a < 3.0:
            out[b] = a * ref[b] + c
        else:
            out[b] = ref[b] - x.mean() + y.mean()
    return out


def _box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=cv2.BORDER_REFLECT)


def match_reference(cloudy: np.ndarray, mask: np.ndarray, ref: np.ndarray, local: bool = True,
                    scales=(96, 24), k: float = 0.25) -> np.ndarray:
    clear = mask == 0
    if clear.sum() < 200:
        return ref.copy()
    out = _global_match(cloudy, clear, ref)
    if local:
        w_clear = clear.astype(np.float32)
        for b in range(3):
            resid = (cloudy[b] - out[b]).astype(np.float32) * w_clear
            off = np.zeros_like(resid)
            for r in scales:
                n = _box(w_clear, r)
                frac = n / (2 * r + 1) ** 2
                off += frac / (frac + k) * _box(resid - off * w_clear, r) / np.maximum(n, 1)
            out[b] = out[b] + cv2.GaussianBlur(off, (0, 0), 4)
    return np.clip(out, 0, 1).astype(np.float32)


class Normaliser:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        if torch.is_tensor(x):
            return x * torch.as_tensor(self.std, device=x.device) + torch.as_tensor(self.mean, device=x.device)
        return x * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_condition(cloudy, mask, ref, norm: Normaliser, ref_present: bool) -> np.ndarray:
    c = norm(cloudy)
    if ref_present and ref is not None:
        r = norm(match_reference(cloudy, mask, ref))
        flag = np.ones((1,) + mask.shape, np.float32)
    else:
        r = np.zeros_like(c)
        flag = np.zeros((1,) + mask.shape, np.float32)
    return np.concatenate([c, mask_channels(mask), r, flag]).astype(np.float32)


def augment(arrs, rng):
    k, flip = rng.integers(4), rng.random() < 0.5
    out = []
    for a in arrs:
        a = np.rot90(a, k, axes=(-2, -1))
        out.append(np.ascontiguousarray(a[..., ::-1] if flip else a))
    return out


def fake_coarse_reference(x: np.ndarray, rng) -> np.ndarray:
    _, h, w = x.shape
    low = np.stack([cv2.resize(cv2.resize(b, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h),
                               interpolation=cv2.INTER_LINEAR) for b in x])
    change = 1 + 0.35 * (np.stack([fractal_noise(h, w, rng, start=2) for _ in range(3)]) - 0.5)
    dx, dy = rng.uniform(-2, 2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    low = np.stack([cv2.warpAffine(b, M, (w, h), borderMode=cv2.BORDER_REFLECT) for b in low])
    gain = rng.uniform(0.85, 1.15, (3, 1, 1)).astype(np.float32)
    return np.clip(low * change * gain, 0, 1).astype(np.float32)


def jitter_mask(mask: np.ndarray, rng, p=0.5, max_px=4) -> np.ndarray:
    if rng.random() >= p or not mask.any():
        return mask
    r = int(rng.integers(1, max_px + 1))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    op = cv2.dilate if rng.random() < 0.5 else cv2.erode
    out = np.zeros_like(mask)
    for v in (2, 1):
        out[op((mask == v).astype(np.uint8), k) > 0] = v
    return out


def perturb_reference(ref: np.ndarray, rng, p=0.5) -> np.ndarray:
    if rng.random() >= p:
        return ref
    _, h, w = ref.shape
    dx, dy = rng.uniform(-1.5, 1.5, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = np.stack([cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for b in ref])
    change = 1 + rng.uniform(0.1, 0.3) * (np.stack([fractal_noise(h, w, rng, start=2) for _ in range(3)]) - 0.5)
    return np.clip(out * change, 0, 1).astype(np.float32)


class SyntheticCloudDataset(Dataset):

    def __init__(self, sources, norm: Normaliser, synth: CloudSynthesiser, patch=128, length=100_000,
                 ref_dropout=0.3, seed=0, jitter=True, robust=False):
        self.sources, self.norm, self.synth = sources, norm, synth
        self.patch, self.length, self.ref_dropout, self.seed, self.jitter = patch, length, ref_dropout, seed, jitter
        self.robust = robust
        self.weights = np.array([s["clear"].shape[-1] * s["clear"].shape[-2] for s in sources], float)
        self.weights /= self.weights.sum()

    def __len__(self):
        return self.length

    def sample(self, rng):
        src = self.sources[rng.choice(len(self.sources), p=self.weights)]
        clear = src["clear"]
        n, _, H, W = clear.shape
        P = self.patch
        y, x = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
        t = rng.integers(n)
        gt = clear[t, :, y:y + P, x:x + P].astype(np.float32)
        if n > 1:
            r = rng.choice([i for i in range(n) if i != t])
            ref = clear[r, :, y:y + P, x:x + P].astype(np.float32)
        else:
            ref = fake_coarse_reference(gt, rng)
        gt, ref = augment([gt, ref], rng)
        if self.jitter:
            gt = np.clip(gt * rng.uniform(0.95, 1.05, (3, 1, 1)), 0, 1).astype(np.float32)
        cloudy, mask, _ = self.synth(gt, rng)
        if self.robust:
            mask, ref = jitter_mask(mask, rng), perturb_reference(ref, rng, 0.3)
        ref_present = rng.random() >= self.ref_dropout
        cond = make_condition(cloudy, mask, ref, self.norm, ref_present)
        return cond, self.norm(gt).astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2 ** 31))
        cond, gt = self.sample(rng)
        return torch.from_numpy(cond), torch.from_numpy(gt)


class FixedSet(Dataset):

    def __init__(self, path, with_valid=False, rematch: Normaliser | None = None):
        d = np.load(path)
        self.cond, self.gt = d["cond"], d["gt"]
        if rematch is not None:
            self.cond = self.cond.astype(np.float32)
            for c in self.cond:
                if c[8].mean() > 0.5:
                    mask = (c[3] > 0.5) * 1 + (c[4] > 0.5) * 2
                    c[5:8] = rematch(match_reference(rematch.inverse(c[:3]), mask.astype(np.uint8),
                                                     rematch.inverse(c[5:8])))
        self.valid = d["valid"] if "valid" in d else np.ones((len(self.gt),) + self.gt.shape[2:], bool)
        self.with_valid = with_valid
        self.meta = json.loads(str(d["meta"])) if "meta" in d else {}

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, i):
        out = (torch.from_numpy(self.cond[i].astype(np.float32)), torch.from_numpy(self.gt[i].astype(np.float32)))
        if self.with_valid:
            out += (torch.from_numpy(self.valid[i].astype(np.float32)),)
        return out


class RealPairDataset(Dataset):

    def __init__(self, path, norm: Normaliser, patch=128, length=100_000, ref_dropout=0.3, seed=0, robust=False):
        d = np.load(path)
        self.cloudy, self.mask, self.truth = d["cloudy"], d["mask"], d["truth"]
        self.valid, self.ref = d["valid"], d["ref"]
        self.norm, self.patch, self.length, self.ref_dropout, self.seed = norm, patch, length, ref_dropout, seed
        self.robust = robust

    def __len__(self):
        return self.length

    def sample(self, rng):
        P = self.patch
        for _ in range(30):
            i = rng.integers(len(self.cloudy))
            H, W = self.mask.shape[1:]
            y, x = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
            m = self.mask[i, y:y + P, x:x + P]
            v = self.valid[i, y:y + P, x:x + P]
            if 0.05 < (m > 0).mean() < 0.98 and v.mean() > 0.6:
                break
        crop = lambda a: a[i, ..., y:y + P, x:x + P].astype(np.float32)
        cloudy, truth, ref, m, v = augment([crop(self.cloudy), crop(self.truth), crop(self.ref),
                                            m.astype(np.float32), v.astype(np.float32)], rng)
        if self.robust:
            m = jitter_mask(m.astype(np.uint8), rng, p=0.3, max_px=2)
        cond = make_condition(cloudy, m.astype(np.uint8), ref, self.norm, rng.random() >= self.ref_dropout)
        return cond, self.norm(truth).astype(np.float32), v

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2 ** 31))
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in self.sample(rng))


class MixedDataset(Dataset):

    def __init__(self, real: RealPairDataset, synth: SyntheticCloudDataset, p_real=0.5, length=100_000, seed=0):
        self.real, self.synth, self.p_real, self.length, self.seed = real, synth, p_real, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2 ** 31))
        if rng.random() < self.p_real:
            out = self.real.sample(rng)
        else:
            c, g = self.synth.sample(rng)
            out = (c, g, np.ones(g.shape[1:], np.float32))
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in out)


class MultiMix(Dataset):

    def __init__(self, parts, length=100_000, seed=0):
        self.sets = [p[0] for p in parts]
        self.p = np.array([p[1] for p in parts], float) / sum(p[1] for p in parts)
        self.length, self.seed = length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2 ** 31))
        out = self.sets[rng.choice(len(self.sets), p=self.p)].sample(rng)
        if len(out) == 2:
            out = (*out, np.ones(out[1].shape[1:], np.float32))
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in out)
