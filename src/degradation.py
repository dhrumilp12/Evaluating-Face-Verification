"""Controlled probe resolution reduction on unstandardized RGB face crops."""
import numpy as np
from PIL import Image

RESOLUTIONS = (160, 80, 40, 20)
RESOLUTION_CONFIG = {"levels": list(RESOLUTIONS), "downsampling": "Pillow BOX",
                     "upsampling": "Pillow BILINEAR", "channel_mode": "float32 F",
                     "output_size": 160, "scope": "probe only, after fixed crop and before standardization"}


def reduce_resolution(raw_crop, size):
    raw = np.asarray(raw_crop, dtype=np.float32)
    if raw.shape != (3, 160, 160) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
        raise ValueError("Expected finite unstandardized 3x160x160 crop in [0,255]")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}")
    if size == 160:
        return raw.copy()
    # Float-mode channels avoid an extra uint8 quantization step.
    output = []
    for channel in raw:
        image = Image.fromarray(channel)
        image = image.resize((size, size), resample=Image.Resampling.BOX)
        image = image.resize((160, 160), resample=Image.Resampling.BILINEAR)
        output.append(np.asarray(image, dtype=np.float32))
    return np.clip(np.stack(output), 0, 255)
