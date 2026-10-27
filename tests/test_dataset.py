import tempfile
import unittest
from pathlib import Path

import numpy as np

from meghmukt.dataset import Normaliser, RealPairDataset, SyntheticCloudDataset, make_condition, match_reference
from meghmukt.synth import CloudSynthesiser
from tests.fixtures import ground

NORM = Normaliser([0.14, 0.12, 0.2], [0.03, 0.04, 0.07])


class TestDataset(unittest.TestCase):
    def test_match_reference_recovers_linear_change(self):
        ref = ground(seed=1)
        cloudy = 0.8 * ref + 0.03
        m = np.zeros(ref.shape[1:], np.uint8)
        np.testing.assert_allclose(match_reference(cloudy, m, ref), cloudy, atol=1e-4)

    def test_condition_layout(self):
        g = ground()
        m = np.zeros(g.shape[1:], np.uint8)
        c = make_condition(g, m, g, NORM, True)
        self.assertEqual(c.shape, (9,) + g.shape[1:])
        self.assertTrue(np.all(c[8] == 1))
        c0 = make_condition(g, m, None, NORM, False)
        self.assertTrue(np.all(c0[5:] == 0))

    def test_synthetic_dataset_sample(self):
        src = [{"clear": np.stack([ground(256, 256, 1), ground(256, 256, 2)])}]
        ds = SyntheticCloudDataset(src, NORM, CloudSynthesiser(), patch=128, length=4)
        cond, gt = ds[0]
        self.assertEqual(tuple(cond.shape), (9, 128, 128))
        self.assertEqual(tuple(gt.shape), (3, 128, 128))

    def test_real_pair_dataset_sample(self):
        g = ground(256, 256)
        mask = np.zeros((1, 256, 256), np.uint8)
        mask[0, 50:150, 50:150] = 1
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pairs.npz"
            np.savez(p, cloudy=g[None], mask=mask, truth=g[None], valid=np.ones((1, 256, 256), bool), ref=g[None])
            cond, gt, valid = RealPairDataset(p, NORM, patch=128, length=2)[0]
        self.assertEqual(tuple(cond.shape), (9, 128, 128))
        self.assertEqual(tuple(valid.shape), (128, 128))


if __name__ == "__main__":
    unittest.main()
