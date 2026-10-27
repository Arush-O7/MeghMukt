import unittest

import cv2
import numpy as np

from meghmukt.metrics import evaluate, psnr, sam_deg, uncertainty_quality
from meghmukt.perception import NIQE, patch_features, sharpness_ratio
from tests.fixtures import ground


class TestMetrics(unittest.TestCase):
    def test_identical_images(self):
        g = ground()
        self.assertGreater(psnr(g, g), 100)
        self.assertLess(sam_deg(g, g), 1e-3)
        self.assertAlmostEqual(sharpness_ratio(g, g, np.ones(g.shape[1:], bool)), 1.0, places=5)

    def test_cloud_and_clear_scored_separately(self):
        g = ground()
        p = g.copy()
        region = np.zeros(g.shape[1:], bool)
        region[40:80, 40:80] = True
        p[:, region] += 0.05
        m = evaluate(p, g, region)
        self.assertLess(m["psnr_cloud"], m["psnr_clear"])
        self.assertGreater(m["psnr_clear"], 100)
        self.assertAlmostEqual(m["cloud_fraction"], region.mean())

    def test_blur_lowers_sharpness_and_raises_niqe(self):
        imgs = [ground(256, 256, s) for s in range(4)]
        nq = NIQE.fit(imgs)
        blur = [np.stack([cv2.GaussianBlur(b, (0, 0), 2) for b in x]) for x in imgs]
        sharp = nq.score(np.concatenate([patch_features(x) for x in imgs]))
        soft = nq.score(np.concatenate([patch_features(x) for x in blur]))
        self.assertLess(sharp, soft)
        self.assertLess(sharpness_ratio(blur[0], imgs[0], np.ones((256, 256), bool)), 0.9)

    def test_uncertainty_quality_detects_informative_map(self):
        rng = np.random.default_rng(0)
        err = rng.random((64, 64))
        q = uncertainty_quality(err + 0.01 * rng.random((64, 64)), err, np.ones((64, 64), bool))
        self.assertGreater(q["unc_spearman"], 0.9)


if __name__ == "__main__":
    unittest.main()
