import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch

from meghmukt.dataset import Normaliser
from meghmukt.inference import export_analysis_ready, reconstruct_scene
from meghmukt.masking import CLOUD
from meghmukt.models import ClassicalFill, build
from tests.fixtures import ground, profile

NORM = Normaliser([0.14, 0.12, 0.2], [0.03, 0.04, 0.07])


class TestInference(unittest.TestCase):
    def setUp(self):
        self.ref = ground(300, 340, seed=1)
        self.img = ground(300, 340, seed=2)
        self.mask = np.zeros((300, 340), np.uint8)
        self.mask[100:180, 150:260] = CLOUD
        self.img[:, self.mask == CLOUD] = 0.5

    def test_tiled_scene_shapes_and_untouched_tiles(self):
        torch.manual_seed(0)
        for model in (ClassicalFill(), build("meghgan").eval()):
            out, unc, conf = reconstruct_scene(model, self.img, self.mask, self.ref, NORM, torch.device("cpu"),
                                               tile=128, overlap=32, tta=False)
            self.assertEqual(out.shape, self.img.shape)
            self.assertEqual(conf.shape, self.mask.shape)
            np.testing.assert_array_equal(out[:, :40, :40], self.img[:, :40, :40])
            self.assertTrue(np.all(conf[self.mask == 0] == 1.0))
            self.assertTrue(np.all((conf >= 0) & (conf <= 1)))

    def test_export_keeps_georeferencing(self):
        out, _, conf = reconstruct_scene(ClassicalFill(), self.img, self.mask, self.ref, NORM, torch.device("cpu"),
                                         tile=128, overlap=32, tta=False)
        prof = profile(300, 340)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.tif"
            export_analysis_ready(p, out, self.mask, conf, prof, "test")
            with rasterio.open(p) as s:
                self.assertEqual(s.count, 5)
                self.assertEqual(s.crs.to_string(), prof["crs"])
                self.assertEqual(s.transform, prof["transform"])
                self.assertEqual(s.descriptions, ("green", "red", "nir", "quality_mask", "confidence_pct"))
                np.testing.assert_array_equal(s.read(4), self.mask)
                self.assertLessEqual(int(s.read(5).max()), 100)


if __name__ == "__main__":
    unittest.main()
