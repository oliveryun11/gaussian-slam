"""Dataset-free CUDA checks for the tile rasterizer's sampled mode.

Build the vendored extension and run from the repository root with
python -m unittest discover -s tests -v.
"""

import unittest

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from utils.recon_helpers import setup_camera


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SampledRasterizerTest(unittest.TestCase):
    def setUp(self):
        # Partial tiles exercise the bounds checks along both image edges.
        self.width, self.height = 37, 35
        camera = setup_camera(
            self.width, self.height,
            [[32, 0, 18], [0, 32, 17], [0, 0, 1]], np.eye(4),
        )
        self.renderer = GaussianRasterizer(
            camera._replace(bg=torch.tensor([0.3, 0.2, 0.1], device="cuda"))
        )
        self.generator = torch.Generator(device="cuda").manual_seed(20)
        means = self.random(64, 3) * 0.7 - 0.35
        means[:, 2] += 2
        self.inputs = {
            "means3D": means,
            "means2D": torch.zeros_like(means),
            "colors_precomp": self.random(64, 3),
            "scales": 0.06 + 0.12 * self.random(64, 3),
            "rotations": torch.nn.functional.normalize(self.random(64, 4), dim=1),
            "opacities": 0.2 + 0.5 * self.random(64, 1),
        }
        for tensor in self.inputs.values():
            tensor.requires_grad_()

    def random(self, *shape):
        return torch.rand(shape, device="cuda", generator=self.generator)

    def test_sampled_outputs_and_gradients_match_full_render(self):
        full_color, full_radii, full_depth = self.renderer(**self.inputs)
        upstream = 2 * self.random(*full_color.shape) - 1
        sparse = self.random(self.width, self.height).T < 0.1
        sparse[:16, :16] = False  # Leave a full tile inactive.
        cases = {
            "all selected": torch.ones_like(sparse),
            "sparse noncontiguous": sparse,
            "none selected": torch.zeros_like(sparse),
        }

        for name, mask in cases.items():
            with self.subTest(mask=name):
                color, radii, depth = self.renderer(**self.inputs, pixel_mask=mask)
                torch.testing.assert_close(color, full_color.masked_fill(~mask, 0), rtol=0, atol=0)
                torch.testing.assert_close(depth, full_depth.masked_fill(~mask, 0), rtol=0, atol=0)
                torch.testing.assert_close(radii, full_radii, rtol=0, atol=0)

                tensors = tuple(self.inputs.values())
                expected = torch.autograd.grad(
                    full_color, tensors, upstream.masked_fill(~mask, 0), retain_graph=True
                )
                # Unselected outputs are constant even with nonzero upstream gradients.
                actual = torch.autograd.grad(color, tensors, upstream)
                for key, gradient, reference in zip(self.inputs, actual, expected):
                    with self.subTest(gradient=key):
                        # CUDA atomic accumulation can change floating-point summation order.
                        torch.testing.assert_close(gradient, reference, atol=3e-4, rtol=1e-4)

    def test_rejects_invalid_pixel_masks(self):
        shape = (self.height, self.width)
        invalid_masks = {
            "dtype": torch.ones(shape, device="cuda"),
            "shape": torch.ones(shape[::-1], device="cuda", dtype=torch.bool),
            "device": torch.ones(shape, dtype=torch.bool),
        }
        for name, mask in invalid_masks.items():
            with self.subTest(mask=name):
                with self.assertRaisesRegex(RuntimeError, "pixel_mask must"):
                    self.renderer(**self.inputs, pixel_mask=mask)
