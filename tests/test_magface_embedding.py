"""MagFace preprocessing, geometry, raw-quality, and complete-load regressions."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from magface_embedding import (
    ALIGNMENT_TEMPLATE, MagFaceAligner, MagFaceEmbedder, align_landmarks,
    clean_backbone_state_dict, preprocess_crops, raw_quality_and_units,
    similarity_transform, validate_raw_embeddings,
)


class MagFaceEmbeddingTests(unittest.TestCase):
    def test_preprocessing_matches_author_opencv_bgr_to_tensor(self):
        from torchvision.transforms import ToTensor

        rgb = np.zeros((112, 112, 3), dtype=np.uint8)
        rgb[:, :] = [17, 128, 255]
        rgb[0, 0] = [255, 0, 42]
        actual = preprocess_crops([Image.fromarray(rgb)])[0]
        expected = ToTensor()(rgb[:, :, ::-1].copy()).numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual[0, 1, 1], 1.0)
        self.assertEqual(actual[2, 0, 0], 1.0)
        self.assertTrue(actual.flags.c_contiguous)

    def test_preprocessing_rejects_old_facenet_crops_and_float_standardization(self):
        for crops in [[], [np.zeros((160, 160, 3), dtype=np.uint8)],
                      [np.zeros((112, 112, 3), dtype=np.float32)],
                      [np.zeros((3, 112, 112), dtype=np.uint8)]]:
            with self.subTest(shape=np.shape(crops)), self.assertRaises(ValueError):
                preprocess_crops(crops)

    def test_similarity_recovers_known_rotation_scale_and_translation(self):
        angle, scale = 0.3, 1.7
        linear = scale * np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
        offset = np.array([31, -12])
        source = (ALIGNMENT_TEMPLATE - offset) @ np.linalg.inv(linear).T
        actual = similarity_transform(source)
        np.testing.assert_allclose(actual[:, :2], linear, atol=1e-12)
        np.testing.assert_allclose(actual[:, 2], offset, atol=1e-12)
        np.testing.assert_allclose(source @ actual[:, :2].T + actual[:, 2],
                                   ALIGNMENT_TEMPLATE, atol=1e-12)

    def test_invalid_landmarks_are_not_silently_box_cropped(self):
        for points in [np.zeros((5, 2)), np.zeros((4, 2)),
                       np.full((5, 2), np.nan), np.tile(np.arange(5)[:, None], (1, 2))]:
            with self.subTest(points=points.tolist()), self.assertRaises(ValueError):
                similarity_transform(points)

    def test_alignment_translation_preserves_color_marker_and_black_border(self):
        pixels = np.zeros((112, 112, 3), dtype=np.uint8)
        pixels[60, 50] = [255, 128, 17]
        # Source points are displaced by 10px. The image moves back 10px.
        crop, matrix = align_landmarks(Image.fromarray(pixels), ALIGNMENT_TEMPLATE + [10, 0])
        np.testing.assert_allclose(matrix, [[1, 0, -10], [0, 1, 0]], atol=1e-12)
        np.testing.assert_allclose(np.asarray(crop)[60, 40], [255, 128, 17], atol=1)
        self.assertEqual(crop.size, (112, 112))
        self.assertEqual(crop.mode, "RGB")
        self.assertFalse(np.asarray(crop)[:, 102:].any())

    def test_raw_quality_is_preserved_before_unit_normalization(self):
        raw = np.zeros((2, 512), dtype=np.float32)
        raw[0, :2], raw[1, :2] = [3, 4], [0, 20]
        before = raw.copy()
        quality, unit = raw_quality_and_units(raw)
        np.testing.assert_array_equal(quality, [5, 20])
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1)
        np.testing.assert_array_equal(raw, before)
        for bad in [np.zeros((1, 512)), np.ones((1, 511)),
                    np.full((1, 512), np.nan), np.full((1, 512), np.inf),
                    np.ones((1, 512), dtype=np.complex64), np.ones(512)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_raw_embeddings(bad)

    def test_checkpoint_wrapper_mapping_requires_every_backbone_entry(self):
        expected = {"conv1.weight": np.ones((3, 3)), "features.weight": np.ones(512)}
        for prefix in ["module.features.", "features.module.", "module.features.module.", ""]:
            state = {prefix + key: value for key, value in expected.items()}
            state["module.fc.weight"] = np.ones((1000, 512))
            self.assertEqual(set(clean_backbone_state_dict(state, expected)), set(expected))
        with self.assertRaisesRegex(ValueError, "missing/mismatched"):
            clean_backbone_state_dict({"module.features.conv1.weight": np.ones((2, 3))}, expected)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            clean_backbone_state_dict({**expected, "module.features.features.weight": np.ones(512)}, expected)

    def test_safe_checkpoint_load_and_single_raw_forward_pass(self):
        import torch

        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.full((512,), 20.0))
                self.calls = 0

            def forward(self, tensor):
                self.calls += 1
                return tensor.mean((1, 2, 3))[:, None] * self.scale[None, :]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test.pth"
            model = TinyBackbone()
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            with patch("vendor.magface_iresnet.iresnet100", return_value=model), \
                    patch("torch.load", wraps=torch.load) as load:
                embedder = MagFaceEmbedder(directory, checkpoint=checkpoint)
            self.assertIs(load.call_args.kwargs["weights_only"], True)
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
            crops = [Image.new("RGB", (112, 112), "white"), Image.new("RGB", (112, 112), "red")]
            features = embedder.embed_batch(crops)
            self.assertEqual(model.calls, 1)  # No hidden flip augmentation.
            self.assertEqual(features.shape, (2, 512))
            np.testing.assert_allclose(features[0], 20)
            np.testing.assert_allclose(features[1], 20 / 3, rtol=1e-6)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_alignment_failure_record_and_central_selection(self):
        import torch

        class FakeDetector:
            def detect(self, image, landmarks=False):
                assert landmarks
                boxes = np.array([[0, 0, 30, 30], [20, 20, 200, 200]])
                points = np.stack([ALIGNMENT_TEMPLATE, ALIGNMENT_TEMPLATE + 50])
                return boxes, np.array([0.999, 0.95]), points

        aligner = MagFaceAligner.__new__(MagFaceAligner)
        aligner.torch, aligner.detector = torch, FakeDetector()
        crop, record = aligner.align(Image.new("RGB", (250, 250)))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["probability"], 0.95)
        self.assertEqual(crop.size, (112, 112))
        with patch.object(aligner.detector, "detect", return_value=(None, None, None)):
            crop, record = aligner.align(Image.new("RGB", (250, 250)))
        self.assertIsNone(crop)
        self.assertEqual(record["status"], "no_face_detected")


if __name__ == "__main__":
    unittest.main()
