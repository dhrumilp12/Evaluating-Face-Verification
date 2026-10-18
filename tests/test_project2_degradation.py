import unittest

import numpy as np
from PIL import Image

from src.project2_degradation import CONDITIONS, apply_condition


class DegradationTests(unittest.TestCase):
    def test_constant_image_and_brightness_quantization(self):
        crop = Image.fromarray(np.full((112, 112, 3), 100, dtype=np.uint8))
        for condition in CONDITIONS:
            actual = np.asarray(apply_condition(crop, condition))
            expected = round(100 * float(condition.split("_")[1])) if condition.startswith("brightness") else 100
            np.testing.assert_array_equal(actual, expected)

    def test_each_transform_preserves_input_and_shape(self):
        pixels = np.random.default_rng(17).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        crop = Image.fromarray(pixels)
        for condition in CONDITIONS:
            actual = apply_condition(crop, condition)
            self.assertEqual(actual.size, (112, 112))
            self.assertEqual(actual.mode, "RGB")
            np.testing.assert_array_equal(np.asarray(crop), pixels)

    def test_blur_spreads_impulse_symmetrically(self):
        pixels = np.zeros((112, 112, 3), dtype=np.uint8)
        pixels[56, 56] = 255
        result = np.asarray(apply_condition(Image.fromarray(pixels), "blur_0.7"))
        self.assertGreater(int(result[56, 56, 0]), int(result[56, 57, 0]))
        self.assertGreater(int(result[56, 57, 0]), 0)
        self.assertEqual(int(result[56, 55, 0]), int(result[56, 57, 0]))
        self.assertEqual(int(result[55, 56, 0]), int(result[57, 56, 0]))

    def test_rejects_wrong_crop_and_condition(self):
        with self.assertRaises(ValueError):
            apply_condition(Image.new("RGB", (160, 160)), "original")
        with self.assertRaises(ValueError):
            apply_condition(Image.new("RGB", (112, 112)), "resolution_20")


if __name__ == "__main__":
    unittest.main()
