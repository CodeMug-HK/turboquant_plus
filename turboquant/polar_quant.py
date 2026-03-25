"""PolarQuant: Random rotation + optimal scalar quantization.

Algorithm 1 from the TurboQuant paper (AISTATS 2026).

After random rotation, coordinates follow a known Beta distribution (Gaussian in
high d), enabling optimal scalar quantization per coordinate independently.
"""

import numpy as np

from turboquant.codebook import optimal_centroids, nearest_centroid_indices
from turboquant.rotation import random_rotation_dense


class PolarQuant:
    """MSE-optimized vector quantizer via random rotation + scalar quantization.

    Usage:
        pq = PolarQuant(d=128, bit_width=2, seed=42)
        indices = pq.quantize(x)          # x: (d,) or (batch, d)
        x_hat = pq.dequantize(indices)    # reconstructed
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42):
        """
        Args:
            d: Vector dimension (e.g., head_dim of attention).
            bit_width: Bits per coordinate (1, 2, 3, 4).
            seed: Random seed for rotation matrix.
        """
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width

        rng = np.random.default_rng(seed)

        # Random rotation matrix — Haar distributed
        self.rotation = random_rotation_dense(d, rng)

        # Optimal codebook for post-rotation distribution
        self.centroids = optimal_centroids(bit_width, d)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize a vector or batch of vectors.

        Args:
            x: Input vector(s), shape (d,) or (batch, d).

        Returns:
            Integer indices, shape (d,) or (batch, d). Values in [0, 2^bit_width).
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        # Rotate: y = Π @ x.T → (d, batch), then transpose to (batch, d)
        y = (self.rotation @ x.T).T

        # Nearest centroid per coordinate
        indices = nearest_centroid_indices(y, self.centroids)

        return indices[0] if single else indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Dequantize indices back to vectors.

        Args:
            indices: Integer indices, shape (d,) or (batch, d).

        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis, :]

        # Look up centroids
        y_hat = self.centroids[indices]  # (batch, d)

        # Inverse rotation: x̃ = Π^T @ ỹ
        x_hat = (self.rotation.T @ y_hat.T).T

        return x_hat[0] if single else x_hat

    def quantize_and_residual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize and return both indices and the residual error.

        Used by TurboQuant's second stage (QJL on residual).

        Returns:
            (indices, residual) where residual = x - dequantize(indices).
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        residual = x - x_hat
        return indices, residual
