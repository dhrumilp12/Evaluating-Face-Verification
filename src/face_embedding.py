"""Fixed MTCNN crop and VGGFace2-pretrained InceptionResnetV1 embeddings."""
from pathlib import Path
import os
import ssl
import urllib.request

import certifi
import numpy as np

MODEL_URL = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
CONFIG = {
    "model": "InceptionResnetV1", "pretrained": "vggface2", "embedding_size": 512,
    "device": "cpu", "image_size": 160, "margin": 0, "min_face_size": 20,
    "thresholds": [0.6, 0.7, 0.7], "factor": 0.709, "minimum_detection_probability": 0.90,
    "selection": "highest-probability box containing image center",
    "crop": "MTCNN.extract bounding-box crop, PIL bilinear resize; no landmark rotation",
    "standardization": "(RGB pixel value - 127.5) / 128.0",
    "seed": 42,
}


def download_checkpoint(root):
    """Cache the official recognition checkpoint using verified HTTPS."""
    checkpoint = Path(root).resolve() / "models/checkpoints" / MODEL_URL.rsplit("/", 1)[1]
    if checkpoint.is_file():
        return checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".part")
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "BiometricsCourseProject/1.0"})
    print("Downloading VGGFace2 recognition checkpoint ...", flush=True)
    try:
        with urllib.request.urlopen(request, context=context, timeout=60) as response, temporary.open("wb") as output:
            received = 0
            expected_size = response.headers.get("Content-Length")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
            if received == 0 or (expected_size is not None and received != int(expected_size)):
                raise ValueError("Incomplete recognition checkpoint download")
        temporary.replace(checkpoint)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return checkpoint


def select_center_box(boxes, probabilities, width, height, min_probability=0.90):
    """Return the centered target face, or an explicit failure reason."""
    if boxes is None or len(boxes) == 0:
        return None, "no_face_detected"
    boxes = np.asarray(boxes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or probabilities.shape != (len(boxes),):
        raise ValueError("Unexpected detector output dimensions")
    center_x, center_y = width / 2, height / 2
    valid = (np.isfinite(boxes).all(axis=1) & np.isfinite(probabilities)
             & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    centered = valid & (boxes[:, 0] <= center_x) & (boxes[:, 2] >= center_x)
    centered &= (boxes[:, 1] <= center_y) & (boxes[:, 3] >= center_y)
    if not centered.any():
        return None, "no_center_face"
    eligible = centered & (probabilities >= min_probability) & (probabilities <= 1.0)
    if not eligible.any():
        return None, "low_detection_probability"
    indices = np.flatnonzero(eligible)
    index = int(indices[np.argmax(probabilities[indices])])
    return index, "ok"


def validate_embedding(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Expected one finite 512-dimensional embedding")
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embedding is not L2 normalized")
    return vector


def cosine_similarity(first, second):
    first, second = validate_embedding(first), validate_embedding(second)
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


class FaceEmbedder:
    def __init__(self, root):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.torch = torch
        # Keep all downloaded recognition weights in the Git-ignored models folder.
        os.environ["TORCH_HOME"] = str(Path(root).resolve() / "models")
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.detector = MTCNN(
            image_size=CONFIG["image_size"], margin=CONFIG["margin"],
            min_face_size=CONFIG["min_face_size"], thresholds=CONFIG["thresholds"],
            factor=CONFIG["factor"], post_process=False, keep_all=True, device="cpu",
        ).eval()
        # Prepopulate the package cache using the same trusted roots as the LFW downloader.
        self.checkpoint = download_checkpoint(root)
        self.model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to("cpu")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def crop(self, picture):
        """Return unstandardized float32 CHW crop and a detection record."""
        picture = picture.convert("RGB")
        with self.torch.inference_mode():
            boxes, probabilities = self.detector.detect(picture)
            index, status = select_center_box(
                boxes, probabilities, *picture.size,
                min_probability=CONFIG["minimum_detection_probability"],
            )
            record = {"status": status, "detected_faces": 0 if boxes is None else len(boxes)}
            if index is None:
                return None, record
            record.update({"box": [float(n) for n in boxes[index]],
                           "probability": float(probabilities[index])})
            crop = self.detector.extract(picture, np.asarray([boxes[index]]), save_path=None)[0]
        raw = crop.detach().cpu().numpy().astype(np.float32)
        if raw.shape != (3, 160, 160) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
            raise ValueError("Unexpected crop shape or pixel range")
        return raw, record

    def embed(self, raw_crop):
        """Standardize once; later degradations must be applied before this step."""
        raw_crop = np.asarray(raw_crop, dtype=np.float32)
        if raw_crop.shape != (3, 160, 160) or not np.isfinite(raw_crop).all():
            raise ValueError("Expected finite 3x160x160 RGB crop")
        if raw_crop.min() < 0 or raw_crop.max() > 255:
            raise ValueError("Input must contain unstandardized pixels in [0,255]")
        tensor = self.torch.from_numpy(raw_crop.copy()).unsqueeze(0)
        tensor = (tensor - 127.5) / 128.0
        with self.torch.inference_mode():
            output = self.model(tensor)[0].detach().cpu().numpy()
        return validate_embedding(output)
