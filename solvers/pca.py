from benchopt import BaseSolver

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from benchmark_utils.metrics import compression_ratio


class PCACompressor:
    """PCA compressor for GYSELA-style nD fields.

    Each field is flattened into a 2D matrix, projected onto its leading
    ``n_components`` principal components and reconstructed. The trailing
    ``n_feature_axes`` axes form the PCA feature (column) space; the remaining
    leading axes are the samples (rows). For an
    fdistribu[species, x, y, vx, vy] field with ``n_feature_axes=2`` this
    matches rows = species*x*y, columns = vx*vy (and rows = x*y for a rank-4
    fdistribu[x, y, vx, vy]).
    """

    ALLOWED_NORMALISATIONS = {"none", "zscore", "log", "asinh"}

    def __init__(self, n_components=32, normalisation="none", alpha=1e-6,
                 clip_nonnegative=False, n_feature_axes=2, random_state=None):
        self.n_components = int(n_components)
        self.normalisation = str(normalisation).lower()
        self.alpha = float(alpha)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.n_feature_axes = int(n_feature_axes)
        self.random_state = random_state

        if self.normalisation not in self.ALLOWED_NORMALISATIONS:
            raise ValueError(
                f"Unknown normalisation '{self.normalisation}'. "
                f"Expected one of {sorted(self.ALLOWED_NORMALISATIONS)}."
            )

    def _array_to_matrix(self, f):
        """Flatten an nD field into a 2D (samples, features) matrix."""
        # Keep at least one leading sample axis even for low-rank fields.
        split = max(1, f.ndim - self.n_feature_axes)
        n_samples = int(np.prod(f.shape[:split]))
        n_features = int(np.prod(f.shape[split:]))
        return f.reshape(n_samples, n_features)

    def _preprocess(self, X, scaler):
        if self.normalisation == "none":
            return X
        if self.normalisation == "log":
            return np.log10(np.clip(X, 1e-16, None))
        if self.normalisation == "asinh":
            return np.arcsinh(X / self.alpha)
        if self.normalisation == "zscore":
            return scaler.fit_transform(X)
        raise RuntimeError(f"Unhandled normalisation: {self.normalisation}")

    def _inverse_preprocess(self, X, scaler):
        if self.normalisation == "none":
            return X
        if self.normalisation == "log":
            return 10.0 ** X
        if self.normalisation == "asinh":
            return self.alpha * np.sinh(X)
        if self.normalisation == "zscore":
            return X * scaler.scale_ + scaler.mean_
        raise RuntimeError(f"Unhandled normalisation: {self.normalisation}")

    def compress_reconstruct(self, x: np.ndarray):
        """PCA reconstruction of ``x`` plus the number of scalars it stores."""
        x = np.asarray(x)
        original_shape, dtype = x.shape, x.dtype

        X = self._array_to_matrix(x.astype(np.float64, copy=False))

        # n_components cannot exceed min(n_samples, n_features); clamp so the
        # same parameters apply across fields of different shapes.
        n_components = min(self.n_components, *X.shape)

        scaler = StandardScaler() if self.normalisation == "zscore" else None
        X_proc = self._preprocess(X, scaler)

        model = PCA(n_components=n_components, svd_solver="auto",
                    random_state=self.random_state)
        coefficients = model.fit_transform(X_proc)

        X_approx = self._inverse_preprocess(
            model.inverse_transform(coefficients), scaler)

        if self.clip_nonnegative:
            X_approx = np.clip(X_approx, 0.0, None)

        # Stored representation: per-sample coefficients + components + mean
        # (+ z-score scaler stats when used).
        n_stored = coefficients.size + model.components_.size + model.mean_.size
        if scaler is not None:
            n_stored += scaler.mean_.size + scaler.scale_.size

        rec = X_approx.reshape(original_shape).astype(dtype, copy=False)
        return rec, n_stored


class Solver(BaseSolver):

    name = "pca"
    sampling_strategy = "run_once"
    requirements = ["numpy", "pip::scikit-learn"]

    parameters = {
        "n_components": [5],
        "normalisation": ["none"],
        "alpha": [1e-6],
        "clip_nonnegative": [False],
        "n_feature_axes": [2],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def run(self, _):
        compressor = PCACompressor(
            n_components=self.n_components,
            normalisation=self.normalisation,
            alpha=self.alpha,
            clip_nonnegative=self.clip_nonnegative,
            n_feature_axes=self.n_feature_axes,
        )
        self.fields_rec = {}
        n_stored = 0
        for name, arr in self.fields.items():
            rec, stored = compressor.compress_reconstruct(arr)
            self.fields_rec[name] = rec
            n_stored += stored
        self.compression_ratio_ = compression_ratio(self.fields, n_stored)

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec,
                    compression_ratio=self.compression_ratio_)
