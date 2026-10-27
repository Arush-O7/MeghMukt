import unittest

import numpy as np

from meghmukt.bluesynth import BlueSynth


class TestBlueSynth(unittest.TestCase):
    def test_ratio_model_recovers_known_relation(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.02, 0.4, (3, 5000))
        ndvi = (X[2] - X[1]) / (X[2] + X[1])
        blue = X[0] * (0.9 - 0.3 * ndvi)
        m = BlueSynth.fit(X, blue, "ratio")
        err = np.abs(m.predict(X) - blue).mean() / blue.mean()
        self.assertLess(err, 0.02)


if __name__ == "__main__":
    unittest.main()
