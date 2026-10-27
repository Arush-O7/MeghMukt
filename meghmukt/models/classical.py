from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .blocks import composite, soft_mask


def fill_numpy(cloudy: np.ndarray, mask: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    out = cloudy.copy()
    if not mask.any():
        return out
    if reference is not None:
        clear = ~mask
        for b in range(3):
            x, y = reference[b][clear], cloudy[b][clear]
            if x.size > 50 and x.std() > 1e-6:
                a, c = np.polyfit(x, y, 1)
            else:
                a, c = 1.0, float(np.mean(y) - np.mean(reference[b])) if x.size else 0.0
            out[b][mask] = a * reference[b][mask] + c
        return out
    h, w = mask.shape
    f = 4 if min(h, w) >= 64 else 1
    small_m = cv2.resize(mask.astype(np.uint8), (w // f, h // f), interpolation=cv2.INTER_NEAREST)
    for b in range(3):
        small = cv2.resize(cloudy[b].astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA)
        filled = cv2.inpaint(small, small_m, 5, cv2.INPAINT_TELEA)
        out[b][mask] = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)[mask]
    return out


class ClassicalFill(nn.Module):

    def forward(self, cond):
        outs = []
        m_all = (soft_mask(cond, blur=0) > 0.5).cpu().numpy()[:, 0]
        for i in range(cond.shape[0]):
            c = cond[i].detach().cpu().numpy()
            if c[8].mean() > 0.5:
                outs.append(np.where(m_all[i][None], c[5:8], c[:3]))
            else:
                outs.append(fill_numpy(c[:3], m_all[i], None))
        pred = torch.from_numpy(np.stack(outs)).to(cond.device)
        return composite(pred, cond)
