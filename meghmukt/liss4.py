from __future__ import annotations

import json
import math
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

ESUN = {2: 184.95, 3: 155.30, 4: 109.20}
BANDS = (2, 3, 4)


class Scene:
    def __init__(self, zip_path: str | Path):
        self.zip = Path(zip_path).resolve()
        self.base = self.zip.stem
        txt = zipfile.ZipFile(self.zip).read(f"{self.base}/BAND_META.txt").decode("latin1")
        self.meta = {k.strip(): v.strip() for k, v in (l.split("=", 1) for l in txt.splitlines() if "=" in l)}
        self.date = datetime.strptime(self.meta["DateOfPass"], "%d-%b-%Y").date()
        self.sun_elev = float(self.meta["SunElevationAtCenter"])
        self.subscene = self.meta.get("SubScene", "")
        self.satellite = self.meta.get("SatID", "")

    def band_path(self, b: int) -> str:
        return f"/vsizip/{self.zip}/{self.base}/BAND{b}.tif"

    @property
    def profile(self):
        with rasterio.open(self.band_path(3)) as s:
            return s.profile.copy()

    def _earth_sun_distance(self):
        doy = self.date.timetuple().tm_yday
        return 1 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))

    def toa_factor(self, b: int):
        lmin, lmax = float(self.meta[f"B{b}_Lmin"]), float(self.meta[f"B{b}_Lmax"])
        k = math.pi * self._earth_sun_distance() ** 2 / (ESUN[b] * math.sin(math.radians(self.sun_elev)))
        return k * (lmax - lmin) / 1023.0, k * lmin

    def read_toa(self, grid: dict, resampling=Resampling.bilinear):
        out, valid = [], None
        for b in BANDS:
            with rasterio.open(self.band_path(b)) as src, WarpedVRT(
                    src, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                    height=grid["height"], resampling=resampling, src_nodata=0, nodata=0) as vrt:
                dn = vrt.read(1).astype(np.float32)
            gain, off = self.toa_factor(b)
            v = dn > 0
            valid = v if valid is None else valid & v
            out.append(np.where(v, dn * gain + off, 0))
        return np.stack(out).astype(np.float32), valid

    def __repr__(self):
        return f"Scene({self.satellite} {self.date} sub {self.subscene} sun {self.sun_elev:.1f})"


class CrossCal:

    def __init__(self, a, b):
        self.a = np.asarray(a, np.float32)[:, None, None]
        self.b = np.asarray(b, np.float32)[:, None, None]

    def __call__(self, toa):
        return np.clip(self.a * toa + self.b, 0, 1).astype(np.float32)

    def save(self, path, **extra):
        Path(path).write_text(json.dumps({"a": self.a.ravel().tolist(), "b": self.b.ravel().tolist(), **extra}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["a"], d["b"])
