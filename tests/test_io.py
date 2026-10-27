import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

from meghmukt.io import Harmoniser, liss4_dn_to_norm, norm_to_liss4_dn, read_raster, write_raster
from tests.fixtures import CRS, profile


class TestIO(unittest.TestCase):
    def test_geotiff_roundtrip_keeps_georeferencing(self):
        arr = (np.random.default_rng(0).random((3, 40, 50)) * 1000).astype(np.uint16)
        prof = profile(40, 50)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.tif"
            write_raster(p, arr, prof, tags={"MODEL": "t"}, band_names=["green", "red", "nir"])
            back, bp = read_raster(p)
            with rasterio.open(p) as src:
                self.assertEqual(src.crs.to_string(), CRS)
                self.assertEqual(src.transform, prof["transform"])
                self.assertEqual(src.descriptions, ("green", "red", "nir"))
                self.assertEqual(src.tags()["MODEL"], "t")
        np.testing.assert_array_equal(back, arr.astype(np.float32))

    def test_liss4_scale_roundtrip(self):
        dn = np.array([[0, 511, 1023]], np.uint16)
        np.testing.assert_array_equal(norm_to_liss4_dn(liss4_dn_to_norm(dn)), dn)

    def test_harmoniser_inverse(self):
        h = Harmoniser([0.7, 0.75, 0.9], [0.09, 0.07, -0.01])
        refl = np.full((3, 4, 4), 0.2, np.float32)
        np.testing.assert_allclose(h.inverse(h(refl)), refl, atol=1e-6)

    def test_harmoniser_anchor_fit(self):
        rng = np.random.default_rng(1)
        s2g, s2c = rng.normal(0.06, 0.01, (3, 500)), rng.normal(0.5, 0.05, (3, 500))
        h = Harmoniser.fit_anchors(s2g, s2c, 0.8 * s2g + 0.08, 0.8 * s2c + 0.08)
        np.testing.assert_allclose(h.gain.ravel(), 0.8, atol=0.02)
        np.testing.assert_allclose(h.bias.ravel(), 0.08, atol=0.01)


if __name__ == "__main__":
    unittest.main()
