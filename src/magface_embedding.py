"""Pinned MagFace-100 inference and a separately documented LFW alignment.

The authors' QMagFace ``preprocessing/embed.py`` reads BGR pixels with OpenCV,
applies ToTensor (division by 255), and keeps a single unnormalized network
output. We reproduce those inference operations without depending on OpenCV.
Our facenet-pytorch detector and Pillow interpolation are a robustness protocol
choice, not a claim to recreate the authors' pre-aligned benchmark images.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Support both ``python -m src...`` and the existing direct-src tests.
    from .face_embedding import select_center_box
except ImportError:
    from face_embedding import select_center_box

MAGFACE_COMMIT = "99bae614ac2643b9694bf18e0c5645272ff6acfa"
QMAGFACE_COMMIT = "395797145da1b1b263cbe7fb1564c6ea91f55b23"
CHECKPOINT_RELATIVE_PATH = "models/project2/magface_epoch_00025.pth"
# Left eye, right eye, nose, left mouth, right mouth, in image coordinates.
ALIGNMENT_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float64)
CONFIG = {
    "model": "MagFace iResNet100", "embedding_size": 512,
    "training_dataset": "MS1MV2", "image_size": 112,
    "magface_commit": MAGFACE_COMMIT, "qmagface_commit": QMAGFACE_COMMIT,
    "channel_order": "BGR", "standardization": "uint8 pixels / 255.0",
    "flip_augmentation": False, "output": "raw final BatchNorm features; no L2 normalization",
    "detector": "facenet_pytorch.MTCNN", "min_face_size": 20,
    "thresholds": [0.6, 0.7, 0.7], "factor": 0.709,
    "minimum_detection_probability": 0.90,
    "selection": "highest-probability box containing image center",
    "alignment": "five-point least-squares similarity to ArcFace 112px template",
    "interpolation": "Pillow affine bilinear; integer pixel-center convention; black fill",
    "alignment_template": ALIGNMENT_TEMPLATE.tolist(), "seed": 42,
}


def similarity_transform(landmarks, target=ALIGNMENT_TEMPLATE):
    """Return the 2x3 orientation-preserving similarity mapping source to target."""
    source, target = np.asarray(landmarks, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("Expected five 2D source and target landmarks")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Landmarks must be finite")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("Degenerate source landmarks")
    # x' = a*x-b*y+tx; y' = b*x+a*y+ty. This is the same similarity
    # least-squares objective used by skimage's upstream alignment.
    system = np.zeros((10, 4), dtype=np.float64)
    system[0::2, 0] = source[:, 0]
    system[0::2, 1] = -source[:, 1]
    system[0::2, 2] = 1
    system[1::2, 0] = source[:, 1]
    system[1::2, 1] = source[:, 0]
    system[1::2, 3] = 1
    a, b, tx, ty = np.linalg.lstsq(system, target.reshape(-1), rcond=None)[0]
    if not np.isfinite([a, b, tx, ty]).all() or a * a + b * b < 1e-12:
        raise ValueError("Degenerate landmark transform")
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def align_landmarks(picture, landmarks):
    """Align an RGB PIL image using a fixed 112px template and black border."""
    if not isinstance(picture, Image.Image):
        raise ValueError("Alignment input must be a PIL image")
    forward = similarity_transform(landmarks)
    inverse = np.linalg.inv(np.vstack([forward, [0.0, 0.0, 1.0]]))[:2]
    # Pillow's affine API locates pixel centers at n+0.5; detector landmarks
    # and the ArcFace template locate them at n. Adjust the inverse offset.
    inverse[:, 2] += 0.5 - inverse[:, :2] @ np.array([0.5, 0.5])
    aligned = picture.convert("RGB").transform(
        (112, 112), Image.Transform.AFFINE, tuple(inverse.reshape(-1)),
        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
    )
    return aligned, forward


def preprocess_crops(crops):
    """Convert RGB uint8 PIL/HWC crops to contiguous float32 BGR NCHW /255."""
    items = list(crops)
    if not items:
        raise ValueError("Expected at least one aligned crop")
    arrays = []
    for crop in items:
        array = np.asarray(crop.convert("RGB") if isinstance(crop, Image.Image) else crop)
        if array.shape != (112, 112, 3) or array.dtype != np.uint8:
            raise ValueError("Expected a 112x112 RGB uint8 crop")
        arrays.append(array[:, :, ::-1].transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(arrays), dtype=np.float32) / np.float32(255.0)


def validate_raw_embeddings(features):
    """Check raw 512D network output without discarding its quality magnitude."""
    raw = np.asarray(features)
    if (raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != 512
            or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)):
        raise ValueError("Expected a nonempty Nx512 real embedding matrix")
    raw = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(raw).all() or np.any(np.linalg.norm(raw.astype(np.float64), axis=1) == 0):
        raise ValueError("Raw embeddings must be finite and nonzero")
    return raw


def raw_quality_and_units(features):
    """Return raw magnitudes and unit vectors, preserving the supplied raw data."""
    raw = validate_raw_embeddings(features)
    quality = np.linalg.norm(raw.astype(np.float64), axis=1)
    return quality, raw.astype(np.float64) / quality[:, None]


def clean_backbone_state_dict(state_dict, expected):
    """Resolve documented upstream wrappers and reject incomplete checkpoints.

    Training classifier tensors may be extra, but every inference state entry,
    including BatchNorm buffers, must have an exactly matching shape. Unlike
    permissive partial loads, this can never leave random backbone weights.
    """
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    cleaned = {}
    prefixes = ("module.features.module.", "features.module.", "module.features.",
                "features.", "module.", "")
    for key, value in state_dict.items():
        if not isinstance(key, str) or not hasattr(value, "shape"):
            continue
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            candidate = key[len(prefix):]
            if candidate in expected and tuple(value.shape) == tuple(expected[candidate].shape):
                if candidate in cleaned:
                    raise ValueError(f"Duplicate checkpoint tensor for {candidate}")
                cleaned[candidate] = value
                break
    missing = sorted(set(expected) - set(cleaned))
    if missing:
        raise ValueError(f"Checkpoint does not completely match iResNet100; missing/mismatched: {missing[:8]}")
    return cleaned


class MagFaceAligner:
    """Detection/alignment can be cached before the recognition weights arrive."""

    def __init__(self, root=None, *, device="cpu", threads=2):
        import torch
        from facenet_pytorch import MTCNN

        if threads < 1:
            raise ValueError("threads must be positive")
        self.torch = torch
        torch.set_num_threads(threads)
        self.detector = MTCNN(
            image_size=112, margin=0, min_face_size=CONFIG["min_face_size"],
            thresholds=CONFIG["thresholds"], factor=CONFIG["factor"],
            post_process=False, keep_all=True, device=device,
        ).eval()

    def align(self, picture):
        picture = picture.convert("RGB")
        with self.torch.inference_mode():
            boxes, probabilities, landmarks = self.detector.detect(picture, landmarks=True)
        index, status = select_center_box(
            boxes, probabilities, *picture.size,
            min_probability=CONFIG["minimum_detection_probability"],
        )
        record = {"status": status, "detected_faces": 0 if boxes is None else len(boxes)}
        if index is None:
            return None, record
        record.update({"box": [float(n) for n in boxes[index]],
                       "probability": float(probabilities[index])})
        if landmarks is None:
            record["status"] = "invalid_landmarks"
            return None, record
        try:
            points = np.asarray(landmarks[index], dtype=np.float64)
            crop, forward = align_landmarks(picture, points)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            record["status"] = "invalid_landmarks"
            return None, record
        record.update({"landmarks": points.tolist(), "transform": forward.tolist()})
        return crop, record


class MagFaceEmbedder:
    """Raw author-checkpoint inference; alignment is loaded only when requested."""

    def __init__(self, root, checkpoint=None, *, device="cpu", threads=2):
        import torch
        try:
            from .vendor.magface_iresnet import iresnet100
        except ImportError:
            from vendor.magface_iresnet import iresnet100

        if threads < 1:
            raise ValueError("threads must be positive")
        self.root, self.torch, self.device, self.threads = Path(root).resolve(), torch, device, threads
        self.checkpoint = Path(checkpoint) if checkpoint is not None else self.root / CHECKPOINT_RELATIVE_PATH
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Download the documented official checkpoint first: {self.checkpoint}")
        torch.manual_seed(CONFIG["seed"])
        torch.set_num_threads(threads)
        self.model = iresnet100(pretrained=False, num_classes=512)
        # Never fall back to unsafe pickle loading of downloaded checkpoints.
        checkpoint_data = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint_data.get("state_dict", checkpoint_data)
        self.model.load_state_dict(clean_backbone_state_dict(state, self.model.state_dict()), strict=True)
        del checkpoint_data, state
        self.model.eval().to(device)
        self.model.requires_grad_(False)
        self._aligner = None

    def align(self, picture):
        if self._aligner is None:
            self._aligner = MagFaceAligner(threads=self.threads)
        return self._aligner.align(picture)

    def embed_batch(self, crops):
        tensor = self.torch.from_numpy(preprocess_crops(crops)).to(self.device)
        with self.torch.inference_mode():
            raw = self.model(tensor).detach().cpu().numpy()
        return validate_raw_embeddings(raw)

    def embed(self, crop):
        return self.embed_batch([crop])[0]
