"""Self-check for the per-solver compression-ratio reporting.

Solver deps (sklearn, pywt) live in benchopt's isolated per-solver envs, so the
compressor tests skip when their dependency is missing.
"""

import numpy as np

from benchmark_utils.metrics import compression_ratio


def test_compression_ratio_helper():
    fields = {"a": np.zeros((4, 5)), "b": np.zeros((2, 2, 2))}  # 20 + 8 = 28
    assert compression_ratio(fields, n_stored=7) == 4.0
    assert compression_ratio(fields, n_stored=0) == 28.0  # guarded /0


def test_pca_stored_count():
    try:
        from solvers.pca import PCACompressor
    except ImportError:
        return "skip (no sklearn)"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 8))
    rec, n_stored = PCACompressor(n_components=3, n_feature_axes=1) \
        .compress_reconstruct(x)
    # coefficients 16*3 + components 3*8 + mean 8 = 80 < 128 original
    assert rec.shape == x.shape
    assert n_stored == 80
    assert compression_ratio({"x": x}, n_stored) > 1.0
    return "ok"


def test_wavelet_quantize_encode():
    try:
        from solvers.wavelet import WaveletCompressor
    except ImportError:
        return "skip (no pywt)"
    # Smooth field: sparse in the wavelet domain, so it entropy-codes small.
    x = np.tile(np.linspace(0, 1, 32), (32, 1)).astype(np.float64)
    rec, n_bytes = WaveletCompressor(quant_step=0.2).compress_reconstruct(x)
    assert rec.shape == x.shape
    assert n_bytes > 0
    assert x.nbytes / n_bytes > 1.0  # actually compressed
    # Coarser quantisation must not produce a larger stream.
    _, fine = WaveletCompressor(quant_step=0.05).compress_reconstruct(x)
    _, coarse = WaveletCompressor(quant_step=1.0).compress_reconstruct(x)
    assert coarse <= fine
    return "ok"


if __name__ == "__main__":
    test_compression_ratio_helper()
    print("helper: ok")
    print("pca:", test_pca_stored_count())
    print("wavelet:", test_wavelet_quantize_encode())
