from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans


def _sel(x, m):
    return x[:, m] if m is not None else x.reshape(x.shape[0], -1)


def psnr(pred, gt, m=None, data_range=1.0):
    mse = np.mean((_sel(pred, m) - _sel(gt, m)) ** 2)
    return float(10 * np.log10(data_range ** 2 / max(mse, 1e-12)))


def mae(pred, gt, m=None):
    return float(np.mean(np.abs(_sel(pred, m) - _sel(gt, m))))


def rmse(pred, gt, m=None):
    return float(np.sqrt(np.mean((_sel(pred, m) - _sel(gt, m)) ** 2)))


def ssim(pred, gt, m=None, data_range=0.5):
    _, smap = structural_similarity(gt, pred, data_range=data_range, channel_axis=0, full=True)
    smap = smap.mean(0)
    return float(smap[m].mean() if m is not None else smap.mean())


def sam_deg(pred, gt, m=None, eps=1e-8):
    p, g = _sel(pred, m).astype(np.float64), _sel(gt, m).astype(np.float64)
    cos = (p * g).sum(0) / np.maximum(np.linalg.norm(p, axis=0) * np.linalg.norm(g, axis=0), eps)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def ndvi(x, eps=1e-6):
    return (x[2] - x[1]) / (x[2] + x[1] + eps)


def ndvi_mae(pred, gt, m=None):
    d = np.abs(ndvi(pred) - ndvi(gt))
    return float(d[m].mean() if m is not None else d.mean())


def _lc_features(x):
    return np.concatenate([x.reshape(3, -1), ndvi(x).reshape(1, -1)]).T


def fit_landcover(gt, k=5, seed=0):
    return KMeans(k, n_init=3, random_state=seed).fit(_lc_features(gt)[::7])


def landcover_agreement(pred, gt, m, km=None):
    if m.sum() < 20:
        return float("nan")
    km = fit_landcover(gt) if km is None else km
    return float((km.predict(_lc_features(gt))[m.ravel()] == km.predict(_lc_features(pred))[m.ravel()]).mean())


def evaluate(pred, gt, region_mask, km=None) -> dict:
    out = {}
    for name, m in (("all", None), ("cloud", region_mask), ("clear", ~region_mask)):
        if m is not None and m.sum() < 20:
            continue
        out.update({f"psnr_{name}": psnr(pred, gt, m), f"ssim_{name}": ssim(pred, gt, m),
                    f"mae_{name}": mae(pred, gt, m), f"rmse_{name}": rmse(pred, gt, m),
                    f"sam_{name}": sam_deg(pred, gt, m)})
    if region_mask.sum() >= 20:
        out["ndvi_mae_cloud"] = ndvi_mae(pred, gt, region_mask)
        out["lulc_agree_cloud"] = landcover_agreement(pred, gt, region_mask, km)
    out["cloud_fraction"] = float(region_mask.mean())
    return out


def uncertainty_quality(unc, err, m) -> dict:
    u, e = unc[m], err[m]
    if u.size < 50:
        return {}
    rho = spearmanr(u, e).statistic
    lo, hi = np.quantile(u, [0.2, 0.8])
    return {"unc_spearman": float(rho), "err_ratio_low_vs_high_conf": float(e[u >= hi].mean() / (e[u <= lo].mean() + 1e-9))}
