import unittest
from pathlib import Path

import numpy as np
import torch

from meghmukt.dataset import Normaliser
from meghmukt.inference import reconstruct_scene
from meghmukt.masking import CLOUD
from meghmukt.models import CONFIDENCE_MODEL, DEFAULT_MODEL, load_checkpoint
from tests.fixtures import ground

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "checkpoints" / f"{DEFAULT_MODEL}.pt").exists(), "trained checkpoints not present")
class TestTrainedPipeline(unittest.TestCase):
    def test_default_model_with_confidence(self):
        dev = torch.device("cpu")
        norm = Normaliser.load(ROOT / "data" / "processed" / "norm.json")
        model = load_checkpoint(ROOT / "checkpoints" / f"{DEFAULT_MODEL}.pt", dev)[0]
        conf_model = load_checkpoint(ROOT / "checkpoints" / f"{CONFIDENCE_MODEL}.pt", dev)[0]
        ref, img = ground(256, 256, 1), ground(256, 256, 1).copy()
        mask = np.zeros((256, 256), np.uint8)
        mask[80:170, 80:170] = CLOUD
        img[:, mask == CLOUD] = 0.5
        out, unc, conf = reconstruct_scene(model, img, mask, ref, norm, dev, tile=256, tta=False, n_samples=2, steps=4,
                                           conf_model=conf_model)
        err_before = np.abs(img - ref)[:, mask == CLOUD].mean()
        err_after = np.abs(out - ref)[:, mask == CLOUD].mean()
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(err_after, 0.2 * err_before)
        self.assertTrue((conf[mask == CLOUD] < 1).any())


if __name__ == "__main__":
    unittest.main()
