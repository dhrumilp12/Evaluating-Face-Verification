"""QMagFace comparison scoring, independently implemented for Project 2.

Reference: Terhorst et al., QMagFace (WACV 2023), Eqs. (3)-(4).
Method checked against arXiv:2111.13475v3 and the authors' scoring code:
https://github.com/pterhoer/QMagFace/blob/main/similarity/qmagface.py

This module does NOT extract embeddings, train a network, fit alpha/beta,
select a decision threshold, or implement an entire benchmark protocol.
Quality values must come from RAW MagFace embedding norms, before L2
normalization. Existing normalized FaceNet embeddings are not a substitute.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Published QMagFace-100 coefficients; NOT universal defaults for other models.
QMAGFACE_100_ALPHA = 0.077428
QMAGFACE_100_BETA = 0.125926


@dataclass(frozen=True)
class PairScores:
    """Separate original cosine scores, adjusted scores, and raw magnitudes."""

    cosine_similarity: FloatArray
    qmagface_score: FloatArray
    reference_quality: FloatArray
    probe_quality: FloatArray


def _real_array(value: Any, name: str, ndim: int) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers, not strings/complex values")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != ndim or not array.size:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    number = float(raw)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite real scalar")
    return number


def raw_features_to_unit_and_quality(
    features: Any, *, allow_unit_norm: bool = False
) -> tuple[FloatArray, FloatArray]:
    """Return unit embeddings and their original, nonzero L2 magnitudes.

    Input has shape (number_of_images, embedding_dimension). Production
    MagFace inference is expected to provide 512-dimensional raw embeddings;
    arbitrary dimensions are supported here to permit small numerical tests.

    Reject an all-unit-norm input by default: that commonly indicates the
    raw quality information was discarded upstream. The explicit override
    is only for documented synthetic tests or ablations. Passing validation
    does not prove that features came from a compatible MagFace checkpoint.
    """
    x = _real_array(features, "features", 2)
    scale = np.max(np.abs(x), axis=1)
    if np.any(scale == 0):
        raise ValueError("Zero-length embeddings have no defined cosine similarity")
    # Scaling avoids overflow while squaring otherwise-valid feature values.
    scaled = x / scale[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        quality = scale * np.sqrt(np.sum(scaled * scaled, axis=1))
    if not np.isfinite(quality).all() or np.any(quality <= 0):
        raise ValueError("Embedding magnitudes must be finite and positive")
    if not allow_unit_norm and np.allclose(quality, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "Embeddings appear unit-normalized; preserve RAW MagFace magnitudes. "
            "FaceNet's normalized cache cannot provide QMagFace qualities."
        )
    return x / quality[:, None], quality


def qmagface_scores(
    cosine_scores: Any,
    reference_quality: Any,
    probe_quality: Any,
    *,
    alpha: float,
    beta: float,
) -> FloatArray:
    """Apply s_hat = s + min(0, beta*s-alpha) * min(q_ref, q_probe).

    All three inputs must be equally shaped, nonempty 1D vectors. Quality
    means raw MagFace embedding magnitude, not detector confidence or a
    match probability. Alpha and beta are explicit to avoid silently using
    the wrong coefficients for a checkpoint.

    Only negligible numerical overshoot in INPUT cosine scores is clipped.
    OUTPUT adjusted scores are not clipped to [-1, 1] and are not sigmoid-
    transformed. Calibrate a separate decision threshold for these scores.
    """
    s = _real_array(cosine_scores, "cosine_scores", 1)
    qr = _real_array(reference_quality, "reference_quality", 1)
    qp = _real_array(probe_quality, "probe_quality", 1)
    if s.shape != qr.shape or s.shape != qp.shape:
        raise ValueError("Scores and both quality vectors must have identical shapes")
    if np.any(qr <= 0) or np.any(qp <= 0):
        raise ValueError("Raw embedding magnitudes must be positive")
    if np.any(s < -1.0 - 1e-7) or np.any(s > 1.0 + 1e-7):
        raise ValueError("Input cosine scores must be in [-1, 1]")
    a = _finite_scalar(alpha, "alpha")
    b = _finite_scalar(beta, "beta")
    s = np.clip(s, -1.0, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        weight = np.minimum(0.0, b * s - a)
        adjusted = s + weight * np.minimum(qr, qp)
    if not np.isfinite(adjusted).all():
        raise ValueError("Parameters and magnitudes produced non-finite scores")
    return adjusted


def score_embedding_pairs(
    reference_features: Any,
    probe_features: Any,
    *,
    alpha: float,
    beta: float,
    allow_unit_norm: bool = False,
) -> PairScores:
    """Score corresponding rows of raw reference/probe embedding matrices.

    The reference and probe may come from different image-quality conditions,
    but must use the SAME checkpoint and compatible preprocessing. Each row
    represents one pair; this function neither constructs nor shuffles pairs.
    """
    ref = _real_array(reference_features, "reference_features", 2)
    probe = _real_array(probe_features, "probe_features", 2)
    if ref.shape != probe.shape:
        raise ValueError("Reference and probe feature matrices must have identical shapes")
    ur, qr = raw_features_to_unit_and_quality(ref, allow_unit_norm=allow_unit_norm)
    up, qp = raw_features_to_unit_and_quality(probe, allow_unit_norm=allow_unit_norm)
    cosine = np.clip(np.einsum("ij,ij->i", ur, up), -1.0, 1.0)
    adjusted = qmagface_scores(cosine, qr, qp, alpha=alpha, beta=beta)
    return PairScores(cosine, adjusted, qr, qp)
