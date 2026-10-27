import io
import unittest

import numpy as np
from PIL import Image

from meghmukt.live import chip_profile
from meghmukt.io import Harmoniser
from meghmukt.webimage import NirFromRGB, load_web_image, to_picture, visible_cloud_mask

LEVELS = {"blue": [0.01, 0.035], "green": [0.023, 0.053], "red": [0.014, 0.041], "nir": [0.064, 0.22]}


def picture(h=96, w=96):
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([60 + 20 * np.sin(x / 7), 90 + 20 * np.cos(y / 9), 50 + 10 * np.sin((x + y) / 11)], -1)
    cloud = (y - h / 2) ** 2 + (x - w / 2) ** 2 < (h / 5) ** 2
    rgb[cloud] = 245
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue(), cloud


class TestWebImage(unittest.TestCase):
    def test_visible_mask_finds_white_cloud(self):
        data, cloud = picture()
        rgb8 = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        m = visible_cloud_mask(rgb8, buffer=0) == 1
        self.assertGreater((m & cloud).sum() / cloud.sum(), 0.9)
        self.assertLess((m & ~cloud).sum() / (~cloud).sum(), 0.02)

    def test_load_web_image_shapes_and_layouts(self):
        rng = np.random.default_rng(0)
        b, g, r = rng.uniform(0.01, 0.2, (3, 2000))
        nir = NirFromRGB.fit(b, g, r, 2 * g + 0.05)
        calib = {"levels": LEVELS, "nir_model": nir.params}
        data, _ = picture()
        for layout in ("natural", "fcc"):
            img, rgb8, info = load_web_image(data, layout, lambda x: x, calib)
            self.assertEqual(img.shape, (3, 96, 96))
            self.assertEqual(rgb8.shape, (96, 96, 3))
            self.assertTrue(np.isfinite(img).all() and (img >= 0).all())

    def test_picture_round_trip_keeps_clear_pixels(self):
        rng = np.random.default_rng(0)
        b, g, r = rng.uniform(0.01, 0.2, (3, 2000))
        calib = {"levels": LEVELS, "nir_model": NirFromRGB.fit(b, g, r, 2 * g + 0.05).params,
                 "cloud_levels": {"blue": 0.5, "green": 0.5, "red": 0.48, "nir": 0.53}}
        harm = Harmoniser(gain=(0.9, 0.95, 0.8), bias=(0.05, 0.04, 0.02))
        data, cloud = picture()
        img, rgb8, info = load_web_image(data, "natural", harm, calib)
        mask = visible_cloud_mask(rgb8)
        back = to_picture(img, mask, rgb8, info, harm)
        self.assertTrue(np.array_equal(back[mask == 0], rgb8[mask == 0]))
        self.assertLess(np.abs(back[..., :2].astype(int) - rgb8[..., :2].astype(int)).max(), 6)

    def test_live_chip_is_centred_and_on_10m_grid(self):
        item = {"properties": {"proj:epsg": 32646}}
        prof = chip_profile(item, 26.15, 91.70, size=512, res=10.0)
        t = prof["transform"]
        self.assertEqual((prof["width"], prof["height"]), (512, 512))
        self.assertEqual((t.c % 10, t.f % 10), (0, 0))
        from rasterio.warp import transform as wt
        x, y = wt("EPSG:4326", "EPSG:32646", [91.70], [26.15])
        self.assertLess(abs(t.c + 2560 - x[0]), 10)
        self.assertLess(abs(t.f - 2560 - y[0]), 10)


if __name__ == "__main__":
    unittest.main()
