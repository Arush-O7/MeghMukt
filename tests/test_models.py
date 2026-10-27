import unittest

import torch

from meghmukt.models import ClassicalFill, anchor, build, composite, soft_mask


def condition(b=2, h=64, w=64, ref=True):
    torch.manual_seed(0)
    c = torch.randn(b, 9, h, w)
    c[:, 3:5] = 0
    c[:, 3, 20:40, 20:40] = 1
    c[:, 8] = 1.0 if ref else 0.0
    if not ref:
        c[:, 5:8] = 0
    return c


class TestModels(unittest.TestCase):
    def test_output_shapes(self):
        c = condition()
        for name in ("unet", "meghgan", "classical"):
            with torch.no_grad():
                self.assertEqual(build(name)(c).shape, (2, 3, 64, 64), name)
        m = build("diffusion").eval()
        self.assertEqual(m.sample(c, steps=3).shape, (2, 3, 64, 64))

    def test_clear_pixels_never_changed(self):
        c = condition()
        far = soft_mask(c)[:, 0] == 0
        for name in ("unet", "meghgan", "classical"):
            with torch.no_grad():
                out = build(name)(c)
            self.assertTrue(torch.allclose(out[:, :, far[0]], c[:, :3][:, :, far[0]]), name)

    def test_anchor_uses_reference_under_mask_only(self):
        c = condition()
        a = anchor(c)
        self.assertTrue(torch.allclose(a[:, :, 30, 30], c[:, 5:8, 30, 30]))
        self.assertTrue(torch.allclose(a[:, :, 2, 2], c[:, :3, 2, 2]))
        nr = condition(ref=False)
        self.assertTrue(torch.allclose(anchor(nr), nr[:, :3]))

    def test_untrained_diffusion_returns_anchor(self):
        c = condition()
        m = build("diffusion").eval()
        self.assertTrue(torch.allclose(m.sample(c, steps=3), composite(anchor(c), c), atol=1e-5))

    def test_classical_pastes_reference(self):
        c = condition()
        out = ClassicalFill()(c)
        self.assertTrue(torch.allclose(out[:, :, 30, 30], c[:, 5:8, 30, 30], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
