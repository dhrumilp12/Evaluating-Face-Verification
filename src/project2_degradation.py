"""Frozen, independent probe transformations for aligned 112-pixel MagFace crops."""
import math

import numpy as np
from PIL import Image


CONDITIONS = (
    "original",
    "resolution_56", "resolution_28", "resolution_14",
    "blur_0.7", "blur_1.4", "blur_2.1",
    "brightness_0.75", "brightness_0.5", "brightness_0.25",
)
MINIMUM_CONDITIONS = ("original", "resolution_14", "blur_2.1", "brightness_0.25")
CONFIG = {
    "crop_size": [112, 112],
    "scope": "probe only; each transform starts from the original aligned crop",
    "resolution": "Pillow bilinear downsample and bilinear restore to 112x112",
    "blur": "separable sampled Gaussian; radius ceil(3*sigma); NumPy reflect padding",
    "brightness": "multiply encoded RGB pixels; no gamma linearization or sensor noise",
    "quantization": "float64 blur/brightness arithmetic; clip [0,255], numpy.rint, uint8",
}


def apply_condition(crop, condition):
    if not isinstance(crop, Image.Image) or crop.mode != "RGB" or crop.size != (112, 112):
        raise ValueError("Expected an RGB PIL image of size 112x112")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "original":
        return crop.copy()
    family, value = condition.split("_", 1)
    if family == "resolution":
        size = int(value)
        return crop.resize((size, size), Image.Resampling.BILINEAR).resize(
            (112, 112), Image.Resampling.BILINEAR)
    values = np.asarray(crop, dtype=np.float64)
    if family == "brightness":
        values = values * float(value)
    else:
        sigma = float(value)
        radius = math.ceil(3 * sigma)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        for axis in (0, 1):
            padding = [(0, 0)] * 3
            padding[axis] = (radius, radius)
            padded = np.pad(values, padding, mode="reflect")
            output = np.zeros_like(values)
            for index, weight in enumerate(kernel):
                region = [slice(None)] * 3
                region[axis] = slice(index, index + 112)
                output += weight * padded[tuple(region)]
            values = output
    return Image.fromarray(np.rint(np.clip(values, 0, 255)).astype(np.uint8))
