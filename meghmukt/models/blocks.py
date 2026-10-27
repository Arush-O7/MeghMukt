from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COND = 9


def soft_mask(cond: torch.Tensor, grow: int = 3, blur: int = 5) -> torch.Tensor:
    m = cond[:, 3:5].amax(1, keepdim=True)
    if grow:
        m = F.max_pool2d(m, 2 * grow + 1, stride=1, padding=grow)
    if blur:
        m = F.avg_pool2d(m, 2 * blur + 1, stride=1, padding=blur, count_include_pad=False)
    return m.clamp(0, 1)


def anchor(cond: torch.Tensor) -> torch.Tensor:
    m = soft_mask(cond) * cond[:, 8:9]
    return cond[:, :3] * (1 - m) + cond[:, 5:8] * m


def composite(pred: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    m = soft_mask(cond)
    return cond[:, :3] * (1 - m) + pred * m


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, temb: int = 0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.t = nn.Linear(temb, cout) if temb else None

    def forward(self, x, t=None):
        h = F.silu(self.n1(self.c1(x)))
        if self.t is not None:
            h = h + self.t(t)[:, :, None, None]
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class AttentionGate(nn.Module):

    def __init__(self, c_skip, c_gate, c_mid):
        super().__init__()
        self.ws = nn.Conv2d(c_skip, c_mid, 1)
        self.wg = nn.Conv2d(c_gate, c_mid, 1)
        self.psi = nn.Conv2d(c_mid, 1, 1)

    def forward(self, skip, gate):
        a = torch.sigmoid(self.psi(F.relu(self.ws(skip) + self.wg(gate))))
        self.last_map = a.detach()
        return skip * a


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch=3, base=32, mults=(1, 2, 4, 8), attention=False, temb=0):
        super().__init__()
        chs = [base * m for m in mults]
        self.temb = temb
        self.inc = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        c = chs[0]
        for co in chs:
            self.down.append(ConvBlock(c, co, temb))
            c = co
        self.mid = ConvBlock(c, c, temb)
        self.up, self.gates = nn.ModuleList(), nn.ModuleList()
        for co in reversed(chs[:-1]):
            self.gates.append(AttentionGate(co, c, co // 2) if attention else nn.Identity())
            self.up.append(ConvBlock(c + co, co, temb))
            c = co
        self.head = nn.Conv2d(c, out_ch, 1)
        nn.init.zeros_(self.head.weight), nn.init.zeros_(self.head.bias)

    def forward(self, x, t=None):
        h = self.inc(x)
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, t)
            if i < len(self.down) - 1:
                skips.append(h)
                h = F.avg_pool2d(h, 2)
        h = self.mid(h, t)
        for gate, blk in zip(self.gates, self.up):
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            if isinstance(gate, AttentionGate):
                s = gate(s, h)
            h = blk(torch.cat([h, s], 1), t)
        return self.head(h)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_ch=3 + 5, base=48):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch, base, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(base, base * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(base * 2, base * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(base * 4, base * 4, 4, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(base * 4, 1, 4, 1, 1)),
        )

    def forward(self, img, cond):
        return self.net(torch.cat([img, cond[:, :5]], 1))
