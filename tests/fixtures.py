import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32646"


def ground(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    g = 0.13 + 0.02 * np.sin(x / 9) + 0.005 * rng.standard_normal((h, w))
    r = 0.11 + 0.02 * np.cos(y / 11) + 0.005 * rng.standard_normal((h, w))
    n = 0.22 + 0.04 * np.sin((x + y) / 13) + 0.005 * rng.standard_normal((h, w))
    river = np.abs(y - 0.5 * x - h / 4) < 4
    g[river], r[river], n[river] = 0.12, 0.09, 0.05
    return np.stack([g, r, n]).astype(np.float32)


def profile(h, w, res=5.0, x0=400000.0, y0=2950000.0):
    return dict(driver="GTiff", crs=CRS, transform=from_origin(x0, y0, res, res), width=w, height=h)


def write_tif(path, arr, prof):
    p = dict(prof, count=arr.shape[0], dtype=arr.dtype.name)
    with rasterio.open(path, "w", **p) as d:
        d.write(arr)
