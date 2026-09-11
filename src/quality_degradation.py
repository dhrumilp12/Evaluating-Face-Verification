"""Independent Gaussian blur and brightness scaling on raw RGB face crops."""
import math
import numpy as np

CONDITIONS = (
    {"condition": "original", "family": "control", "level": 0.0},
    *({"condition": f"blur_sigma_{s:g}", "family": "blur", "level": s} for s in (1.0, 2.0, 3.0)),
    *({"condition": f"brightness_{b:g}", "family": "brightness", "level": b} for b in (0.75, 0.50, 0.25)),
)
QUALITY_CONFIG = {
    "conditions": list(CONDITIONS),
    "blur": "separable sampled Gaussian; radius ceil(3*sigma); normalized kernel; NumPy reflect padding",
    "brightness": "multiply encoded RGB pixel values by factor; no gamma conversion or added noise",
    "arithmetic": "float32 crops, float64 Gaussian kernel/accumulation, float32 output; no uint8 quantization",
    "scope": "each transform starts from original 3x160x160 crop, probe only, before existing standardization",
}


def raw_copy(raw_crop):
    raw = np.asarray(raw_crop, dtype=np.float32)
    if raw.shape != (3, 160, 160) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
        raise ValueError("Expected finite unstandardized 3x160x160 crop in [0,255]")
    return raw.copy()


def gaussian_blur(raw_crop, sigma):
    """Sigma is measured in pixels of the fixed 160px crop; zero is exact identity."""
    raw = raw_copy(raw_crop)
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, (int, float, np.number)) or sigma not in (0, 1, 2, 3):
        raise ValueError("Sigma must be 0, 1, 2, or 3 pixels")
    if sigma == 0:
        return raw
    radius = math.ceil(3 * sigma)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = raw.astype(np.float64)
    for axis in (1, 2):
        padding = [(0, 0)] * 3
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="reflect")
        blurred = np.zeros_like(result)
        for index, weight in enumerate(kernel):
            region = [slice(None)] * 3
            region[axis] = slice(index, index + 160)
            blurred += weight * padded[tuple(region)]
        result = blurred
    return np.clip(result, 0, 255).astype(np.float32)


def reduce_brightness(raw_crop, factor):
    """Scale raw encoded RGB; this is not a physical low-light camera simulator."""
    raw = raw_copy(raw_crop)
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, float, np.number)) or factor not in (1, .75, .5, .25):
        raise ValueError("Brightness factor must be 1, .75, .5, or .25")
    return raw * np.float32(factor)


def apply_condition(raw_crop, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown experimental condition")
    if condition["family"] == "control":
        return raw_copy(raw_crop)
    if condition["family"] == "blur":
        return gaussian_blur(raw_crop, condition["level"])
    return reduce_brightness(raw_crop, condition["level"])
