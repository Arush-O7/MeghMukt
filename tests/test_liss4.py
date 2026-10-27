import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from meghmukt.liss4 import CrossCal, Scene
from tests.fixtures import profile, write_tif

META = """ProductID=1
SatID= IRS-R2
SubScene= B
DateOfPass= 01-NOV-2023
SunElevationAtCenter=  45.0
B2_Lmin=   0.0
B3_Lmin=   0.0
B4_Lmin=   0.0
B2_Lmax=  52.0
B3_Lmax=  47.0
B4_Lmax=  31.5
"""


def fake_product(d):
    base = "TESTSCENE"
    z = Path(d) / f"{base}.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"{base}/BAND_META.txt", META)
        for b in (2, 3, 4):
            p = Path(d) / f"BAND{b}.tif"
            arr = np.full((1, 20, 30), 100 * b, np.uint16)
            arr[0, :, :3] = 0
            write_tif(p, arr, profile(20, 30))
            zf.write(p, f"{base}/BAND{b}.tif")
    return z


class TestLiss4(unittest.TestCase):
    def test_scene_metadata_and_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            sc = Scene(fake_product(d))
            self.assertEqual(str(sc.date), "2023-11-01")
            self.assertEqual(sc.subscene, "B")
            g2, o2 = sc.toa_factor(2)
            self.assertGreater(g2, 0)
            self.assertEqual(o2, 0)
            toa, valid = sc.read_toa(dict(crs=sc.profile["crs"], transform=sc.profile["transform"], width=30, height=20))
            self.assertEqual(toa.shape, (3, 20, 30))
            self.assertFalse(valid[:, :3].any())
            self.assertTrue(valid[:, 3:].all())
            self.assertTrue(np.all(toa[:, valid] > 0) and np.all(toa[:, valid] < 1))

    def test_crosscal_roundtrip(self):
        c = CrossCal([0.74, 1.08, 1.18], [0.07, 0.03, -0.03])
        with tempfile.TemporaryDirectory() as d:
            c.save(Path(d) / "c.json", pair="test")
            c2 = CrossCal.load(Path(d) / "c.json")
        x = np.full((3, 2, 2), 0.1, np.float32)
        np.testing.assert_allclose(c(x), c2(x))


if __name__ == "__main__":
    unittest.main()
