"""Synthetic mathematical checks only; these are not recognition results."""
import math
import unittest

import numpy as np

from src.qmagface_scoring import (
    QMAGFACE_100_ALPHA, QMAGFACE_100_BETA,
    qmagface_scores, raw_features_to_unit_and_quality, score_embedding_pairs,
)


class QMagFaceScoringTests(unittest.TestCase):
    def test_published_100_coefficients(self):
        self.assertEqual(QMAGFACE_100_ALPHA, 0.077428)
        self.assertEqual(QMAGFACE_100_BETA, 0.125926)

    def test_hand_calculated_score(self):
        # 0.5 + min(0, 0.12*0.5 - 0.08)*min(2, 3) = 0.46
        actual = qmagface_scores([0.5], [2.0], [3.0], alpha=0.08, beta=0.12)
        np.testing.assert_allclose(actual, [0.46], atol=1e-14)

    def test_high_similarity_is_unchanged(self):
        actual = qmagface_scores([0.9], [30.0], [40.0],
                                alpha=QMAGFACE_100_ALPHA, beta=QMAGFACE_100_BETA)
        np.testing.assert_array_equal(actual, [0.9])

    def test_weight_switch_boundary(self):
        alpha, beta = 0.08, 0.12
        score = alpha / beta
        actual = qmagface_scores([score], [30.0], [40.0], alpha=alpha, beta=beta)
        np.testing.assert_allclose(actual, [score], rtol=0, atol=1e-14)

    def test_zero_weight_ablation_is_cosine(self):
        s = np.array([-0.8, 0.0, 0.5, 0.9])
        actual = qmagface_scores(s, [2]*4, [4]*4, alpha=0.0, beta=0.0)
        np.testing.assert_array_equal(actual, s)

    def test_uses_minimum_not_mean_quality(self):
        a = qmagface_scores([0.5], [2], [100], alpha=0.08, beta=0.12)
        b = qmagface_scores([0.5], [2], [3], alpha=0.08, beta=0.12)
        np.testing.assert_array_equal(a, b)

    def test_fixed_score_higher_quality_has_larger_penalty(self):
        actual = qmagface_scores([0.5, 0.5], [2, 10], [3, 20], alpha=0.08, beta=0.12)
        self.assertLess(actual[1], actual[0])

    def test_adjustment_does_not_increase_cosine(self):
        s = np.linspace(-1, 1, 101)
        actual = qmagface_scores(s, np.full(101, 30), np.full(101, 40),
                                alpha=QMAGFACE_100_ALPHA, beta=QMAGFACE_100_BETA)
        self.assertTrue(np.all(actual <= s))

    def test_adjusted_output_is_not_clipped(self):
        actual = qmagface_scores([-0.5], [30], [40], alpha=0.08, beta=0.12)
        self.assertLess(actual[0], -1.0)
        self.assertAlmostEqual(actual[0], -4.7)

    def test_quality_is_preserved_before_normalization(self):
        unit, quality = raw_features_to_unit_and_quality([[3, 4], [5, 12]])
        np.testing.assert_allclose(quality, [5, 13])
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), [1, 1])

    def test_inputs_are_not_modified(self):
        ref = np.array([[3., 4.], [5., 12.]])
        probe = np.array([[4., -3.], [12., 5.]])
        copies = ref.copy(), probe.copy()
        score_embedding_pairs(ref, probe, alpha=0.08, beta=0.12)
        np.testing.assert_array_equal(ref, copies[0])
        np.testing.assert_array_equal(probe, copies[1])

    def test_pair_order_is_symmetric(self):
        ref, probe = [[3, 4], [5, 12]], [[4, -3], [12, 5]]
        a = score_embedding_pairs(ref, probe, alpha=0.08, beta=0.12)
        b = score_embedding_pairs(probe, ref, alpha=0.08, beta=0.12)
        np.testing.assert_allclose(a.qmagface_score, b.qmagface_score)
        np.testing.assert_allclose(a.reference_quality, b.probe_quality)

    def test_embedding_and_precomputed_score_interfaces_agree(self):
        result = score_embedding_pairs([[3, 4]], [[4, -3]], alpha=0.08, beta=0.12)
        np.testing.assert_allclose(result.cosine_similarity, [0.0], atol=1e-15)
        expected = qmagface_scores([0], [5], [5], alpha=0.08, beta=0.12)
        np.testing.assert_allclose(result.qmagface_score, expected)

    def test_random_pairs_match_independent_scalar_formula(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(size=(200, 16)) * 3
        probe = rng.normal(size=(200, 16)) * 4
        result = score_embedding_pairs(ref, probe,
                                       alpha=QMAGFACE_100_ALPHA, beta=QMAGFACE_100_BETA)
        expected = []
        for r, p in zip(ref.tolist(), probe.tolist()):
            qr = math.sqrt(sum(v*v for v in r))
            qp = math.sqrt(sum(v*v for v in p))
            s = sum(x*y for x, y in zip(r, p)) / (qr * qp)
            expected.append(s + min(0.0, QMAGFACE_100_BETA*s-QMAGFACE_100_ALPHA)*min(qr, qp))
        np.testing.assert_allclose(result.qmagface_score, expected, rtol=1e-12, atol=1e-12)

    def test_float32_input_gives_float64_output(self):
        result = score_embedding_pairs(np.array([[3, 4]], dtype=np.float32),
                                       np.array([[4, -3]], dtype=np.float32), alpha=0.08, beta=0.12)
        self.assertEqual(result.qmagface_score.dtype, np.dtype("float64"))

    def test_unit_normalized_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unit-normalized"):
            raw_features_to_unit_and_quality([[1, 0], [0, 1]])

    def test_unit_norm_override_is_explicit(self):
        unit, quality = raw_features_to_unit_and_quality([[1, 0]], allow_unit_norm=True)
        np.testing.assert_array_equal(quality, [1])
        np.testing.assert_array_equal(unit, [[1, 0]])

    def test_zero_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            raw_features_to_unit_and_quality([[0, 0]])

    def test_nonfinite_embeddings_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                raw_features_to_unit_and_quality([[value, 2]])

    def test_feature_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            score_embedding_pairs([[2, 3]], [[2, 3, 4]], alpha=0.08, beta=0.12)

    def test_vector_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            qmagface_scores([0.5, 0.6], [2], [3, 4], alpha=0.08, beta=0.12)

    def test_invalid_cosine_is_rejected(self):
        for s in [-1.1, 1.1, np.nan, np.inf]:
            with self.subTest(s=s), self.assertRaises(ValueError):
                qmagface_scores([s], [2], [3], alpha=0.08, beta=0.12)

    def test_negligible_cosine_roundoff_is_handled(self):
        actual = qmagface_scores([1+1e-8], [2], [3], alpha=0.08, beta=0.12)
        np.testing.assert_array_equal(actual, [1])

    def test_invalid_quality_is_rejected(self):
        for q in [0, -1, np.nan, np.inf]:
            for side in ["reference", "probe"]:
                with self.subTest(q=q, side=side), self.assertRaises(ValueError):
                    qmagface_scores([0.5], [q if side == "reference" else 2],
                                    [q if side == "probe" else 3], alpha=0.08, beta=0.12)

    def test_invalid_parameters_are_rejected(self):
        for value in [np.nan, np.inf, [0.1], "0.1", 1j]:
            for name in ["alpha", "beta"]:
                kwargs = dict(alpha=0.08, beta=0.12)
                kwargs[name] = value
                with self.subTest(value=value, name=name), self.assertRaises(ValueError):
                    qmagface_scores([0.5], [2], [3], **kwargs)

    def test_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            qmagface_scores([], [], [], alpha=0.08, beta=0.12)
        with self.assertRaises(ValueError):
            raw_features_to_unit_and_quality(np.empty((0, 512)))

    def test_scalars_and_nonreal_inputs_are_rejected(self):
        for s in [0.5, ["0.5"], [0.5+1j], [True]]:
            with self.subTest(s=s), self.assertRaises(ValueError):
                qmagface_scores(s, [2], [3], alpha=0.08, beta=0.12)

    def test_magnitude_scaling_changes_quality_not_cosine(self):
        ref, probe = np.array([[3., 4.]]), np.array([[4., -3.]])
        a = score_embedding_pairs(ref, probe, alpha=0.08, beta=0.12)
        b = score_embedding_pairs(ref*2, probe*2, alpha=0.08, beta=0.12)
        np.testing.assert_allclose(a.cosine_similarity, b.cosine_similarity)
        np.testing.assert_allclose(b.reference_quality, 2*a.reference_quality)
        self.assertLess(b.qmagface_score[0], a.qmagface_score[0])

    def test_numeric_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            qmagface_scores([-1], [1e308], [1e308], alpha=1e308, beta=1e308)


if __name__ == "__main__":
    unittest.main()
