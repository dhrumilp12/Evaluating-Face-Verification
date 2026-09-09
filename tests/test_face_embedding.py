"""Focused regression checks; run with python -m unittest discover -s tests."""
import io
import json
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from face_embedding import (
    MODEL_URL, cosine_similarity, download_checkpoint, select_center_box,
    validate_embedding,
)
from embedding_smoke_test import run_smoke_test, select_development_pairs


class FaceEmbeddingTests(unittest.TestCase):
    def test_peripheral_face_cannot_displace_central_target(self):
        boxes = [[0, 0, 50, 50], [70, 70, 180, 180], [60, 60, 190, 190]]
        self.assertEqual(select_center_box(boxes, [0.999, 0.95, 0.97], 250, 250), (2, "ok"))

    def test_detection_failures_have_distinct_reasons(self):
        cases = [
            (None, None, "no_face_detected"),
            ([[0, 0, 50, 50]], [0.99], "no_center_face"),
            ([[70, 70, 180, 180]], [0.89], "low_detection_probability"),
            ([[70, 70, 180, 180]], [np.nan], "no_center_face"),
        ]
        for boxes, probabilities, reason in cases:
            with self.subTest(reason=reason, probabilities=probabilities):
                self.assertEqual(select_center_box(boxes, probabilities, 250, 250), (None, reason))
        with self.assertRaises(ValueError):
            select_center_box([[0, 0, 200, 200]], [0.99, 0.98], 250, 250)

    def test_invalid_embeddings_are_rejected(self):
        unit = np.zeros(512, dtype=np.float32)
        unit[0] = 1
        np.testing.assert_array_equal(validate_embedding(unit), unit)
        for vector in [unit[:511], unit * 2, np.zeros(512), np.full(512, np.nan), np.full(512, np.inf)]:
            with self.subTest(shape=vector.shape), self.assertRaises(ValueError):
                validate_embedding(vector)

    def test_cosine_uses_unit_vectors_without_a_decision_threshold(self):
        first, second = np.eye(2, 512, dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(first, first), 1)
        self.assertAlmostEqual(cosine_similarity(first, second), 0)
        self.assertAlmostEqual(cosine_similarity(first, -first), -1)

    def test_subset_is_balanced_and_requires_both_classes(self):
        pairs = [{'pair_index': i, 'label': 1} for i in range(12)]
        pairs += [{'pair_index': 1100 + i, 'label': 0} for i in range(12)]
        selected = select_development_pairs(pairs)
        self.assertEqual([row['pair_index'] for row in selected], list(range(10)) + list(range(1100, 1110)))
        with self.assertRaises(ValueError):
            select_development_pairs(pairs[:12])

    def test_checkpoint_download_verifies_tls_and_reuses_cache(self):
        payload = b'checkpoint fixture'
        response = io.BytesIO(payload)
        response.headers = {'Content-Length': str(len(payload))}
        with tempfile.TemporaryDirectory() as directory:
            with patch('face_embedding.urllib.request.urlopen', return_value=response) as open_url:
                path = download_checkpoint(directory)
                self.assertEqual(path.read_bytes(), payload)
                args, kwargs = open_url.call_args
                self.assertEqual(args[0].full_url, MODEL_URL)
                self.assertEqual(kwargs['context'].verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(kwargs['context'].check_hostname)
            with patch('face_embedding.urllib.request.urlopen') as open_url:
                self.assertEqual(download_checkpoint(directory), path)
                open_url.assert_not_called()

    def test_incomplete_checkpoint_is_not_cached(self):
        response = io.BytesIO(b'incomplete')
        response.headers = {'Content-Length': '100'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('face_embedding.urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    download_checkpoint(directory)
            self.assertEqual(list((Path(directory) / 'models/checkpoints').iterdir()), [])

    def test_failed_dataset_validation_prevents_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'results/metrics/dataset_summary.json'
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'all_checks_passed': False}))
            with patch('embedding_smoke_test.FaceEmbedder') as embedder:
                with self.assertRaisesRegex(RuntimeError, 'dataset checks did not pass'):
                    run_smoke_test(directory)
                embedder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
