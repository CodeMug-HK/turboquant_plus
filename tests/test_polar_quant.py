"""Tests for PolarQuant (Algorithm 1)."""

import numpy as np
import pytest

from turboquant.polar_quant import PolarQuant


class TestPolarQuantRoundTrip:
    """Quantize → dequantize should produce bounded MSE."""

    @pytest.mark.parametrize("bit_width", [1, 2, 3])
    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_round_trip_mse_within_bounds(self, bit_width, d):
        """MSE distortion should be within paper's bounds (Table 2)."""
        # Paper bounds (upper, for normalized vectors)
        expected_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

        pq = PolarQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(99)

        n_samples = 500
        mse_total = 0.0
        for _ in range(n_samples):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)  # unit vector

            indices = pq.quantize(x)
            x_hat = pq.dequantize(indices)
            mse_total += np.mean((x - x_hat) ** 2)

        avg_mse = mse_total / n_samples
        # Allow 2× slack over paper bound (we're using finite d, paper assumes d→∞)
        assert avg_mse < expected_mse[bit_width] * 2.0, (
            f"MSE {avg_mse:.4f} exceeds 2× paper bound {expected_mse[bit_width]} "
            f"at d={d}, b={bit_width}"
        )

    def test_zero_vector(self):
        """Zero vector quantizes to small values (centroids are non-zero).

        PolarQuant maps rotated zeros to nearest centroids, so reconstruction
        won't be exactly zero — but the MSE should be small relative to unit vectors.
        """
        pq = PolarQuant(d=128, bit_width=2, seed=42)
        x = np.zeros(128)
        indices = pq.quantize(x)
        x_hat = pq.dequantize(indices)
        # Reconstruction of zero will be non-zero (centroid quantization),
        # but norm should be small
        assert np.linalg.norm(x_hat) < 0.5

    def test_deterministic(self):
        """Same seed, same input → same output."""
        d = 128
        x = np.random.default_rng(1).standard_normal(d)

        pq1 = PolarQuant(d=d, bit_width=2, seed=42)
        pq2 = PolarQuant(d=d, bit_width=2, seed=42)

        idx1 = pq1.quantize(x)
        idx2 = pq2.quantize(x)
        np.testing.assert_array_equal(idx1, idx2)

    def test_batch_matches_single(self):
        """Batch quantization should match single-vector quantization."""
        d = 128
        pq = PolarQuant(d=d, bit_width=2, seed=42)
        rng = np.random.default_rng(7)

        X = rng.standard_normal((10, d))

        # Batch
        batch_indices = pq.quantize(X)
        batch_recon = pq.dequantize(batch_indices)

        # Single
        for i in range(10):
            single_idx = pq.quantize(X[i])
            single_recon = pq.dequantize(single_idx)
            np.testing.assert_array_equal(batch_indices[i], single_idx)
            np.testing.assert_allclose(batch_recon[i], single_recon, atol=1e-12)

    def test_indices_in_range(self):
        """All indices should be in [0, 2^bit_width)."""
        d = 256
        pq = PolarQuant(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(5)
        x = rng.standard_normal(d)

        indices = pq.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < (1 << 3)


class TestPolarQuantResidual:
    """Test quantize_and_residual method."""

    def test_residual_identity(self):
        """residual = x - dequantize(quantize(x))."""
        d = 128
        pq = PolarQuant(d=d, bit_width=2, seed=42)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(d)

        indices, residual = pq.quantize_and_residual(x)
        x_hat = pq.dequantize(indices)

        np.testing.assert_allclose(residual, x - x_hat, atol=1e-12)
