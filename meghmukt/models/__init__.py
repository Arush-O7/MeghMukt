from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import N_COND, PatchDiscriminator, UNet, anchor, composite, soft_mask
from .classical import ClassicalFill

MODEL_NAMES = {"classical": "M0 Classical temporal fill", "unet": "M1 U-Net baseline",
               "meghgan": "M2 MeghGAN (ours)", "diffusion": "M3 Conditional diffusion",
               "meghgan_real": "M2 MeghGAN + real-cloud training (Sentinel-2 default)",
               "meghgan_liss4": "M2 MeghGAN + real LISS-IV training (LISS-IV default)",
               "diffusion_real": "M3 Diffusion + real-cloud training"}
DEFAULT_MODEL, DEFAULT_MODEL_S2, CONFIDENCE_MODEL = "meghgan_liss4", "meghgan_real", "diffusion_real"


def default_model(sensor: str = "liss4") -> str:
    return DEFAULT_MODEL_S2 if sensor == "s2" else DEFAULT_MODEL


class UNetBaseline(nn.Module):

    def __init__(self, base=32):
        super().__init__()
        self.net = UNet(5, 3, base=base)

    def forward(self, cond):
        return composite(cond[:, :3] + self.net(cond[:, :5]), cond)


class MeghGANGenerator(nn.Module):

    def __init__(self, base=32):
        super().__init__()
        self.net = UNet(N_COND, 3, base=base, attention=True)

    def forward(self, cond):
        return composite(anchor(cond) + self.net(cond), cond)


class ConditionalDiffusion(nn.Module):

    SCALE = 1.0

    def __init__(self, base=32, timesteps=1000):
        super().__init__()
        self.T = timesteps
        self.temb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, self.temb_dim), nn.SiLU(), nn.Linear(self.temb_dim, self.temb_dim))
        self.net = UNet(3 + N_COND, 3, base=base, mults=(1, 2, 4, 4), temb=self.temb_dim)
        s = 0.008
        t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = (ac / ac[0]).clamp(1e-5, 1)
        self.register_buffer("alpha_bar", ac[1:])

    def _temb(self, t):
        half = self.time_mlp[0].in_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.time_mlp(torch.cat([a.sin(), a.cos()], 1))

    def x0(self, xt, t, cond):
        return self.net(torch.cat([xt, cond], 1), self._temb(t))

    def loss(self, target, cond, valid=None):
        x0 = (target - anchor(cond)) * self.SCALE
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        ab = self.alpha_bar[t][:, None, None, None]
        noise = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        w = 1 + 4 * soft_mask(cond)
        if valid is not None:
            w = w * valid
        return ((self.x0(xt, t, cond) - x0) ** 2 * w).mean()

    @torch.no_grad()
    def sample(self, cond, steps=20, generator=None):
        b, _, h, w = cond.shape
        x = torch.randn((b, 3, h, w), device=cond.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=cond.device).long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            x0 = self.x0(x, t.repeat(b), cond).clamp(-4, 4)
            e = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * e
        return composite(anchor(cond) + x / self.SCALE, cond)

    def forward(self, cond, steps=20, generator=None):
        return self.sample(cond, steps, generator)


def build(name: str, cfg: dict | None = None):
    if name == "unet":
        return UNetBaseline()
    if name == "meghgan":
        return MeghGANGenerator()
    if name == "diffusion":
        T = (cfg or {}).get("train", {}).get("diffusion", {}).get("timesteps", 1000)
        return ConditionalDiffusion(timesteps=T)
    if name == "classical":
        return ClassicalFill()
    raise ValueError(name)


def save_checkpoint(path, model, name, norm, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "state": model.state_dict(), "norm": norm, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build(ck["name"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck
