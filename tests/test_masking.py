import unittest

import numpy as np

from meghmukt.masking import CLEAR, CLOUD, SHADOW, MaskParams, coverage, detect, mask_channels, pixel_size, shadow_zone
from meghmukt.synth import CloudSynthesiser
from tests.fixtures import ground, profile


class TestMasking(unittest.TestCase):
    def test_clear_scene_has_no_cloud(self):
        m, _ = detect(ground(), MaskParams(bright=0.24, flatness=0.7))
        self.assertLess(coverage(m)["cloud"], 0.01)

    def test_detects_synthetic_cloud(self):
        g = ground(256, 256)
        cloudy, true, _ = CloudSynthesiser(shadow_prob=0.0)(g, np.random.default_rng(3))
        det, _ = detect(cloudy, MaskParams(bright=0.24, flatness=0.7))
        t, d = true == CLOUD, det == CLOUD
        iou = (t & d).sum() / max((t | d).sum(), 1)
        self.assertGreater(iou, 0.6, f"cloud IoU {iou:.2f}")

    def test_mask_codes_and_channels(self):
        m = np.array([[CLEAR, CLOUD], [SHADOW, CLEAR]], np.uint8)
        ch = mask_channels(m)
        self.assertEqual(ch.shape, (2, 2, 2))
        self.assertEqual(ch[0].sum(), 1)
        self.assertEqual(ch[1].sum(), 1)
        self.assertAlmostEqual(coverage(m)["cloud"], 0.25)

    def test_shadow_zone_none_without_cloud(self):
        self.assertIsNone(shadow_zone(np.zeros((64, 64), bool), np.ones((64, 64)), 1.0, 60))

    def test_pixel_size(self):
        self.assertEqual(pixel_size(profile(10, 10, res=5.0)), 5.0)
        self.assertEqual(pixel_size(None, 10.0), 10.0)


if __name__ == "__main__":
    unittest.main()
