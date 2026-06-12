from benchopt import BaseSolver

import numpy as np
import pywt


class WaveletCompressor:
    def __init__(self, wavelet="db2", compression_ratio=10, level=None,
                 mode="periodic"):
        self.wavelet = wavelet
        self.compression_ratio = compression_ratio
        self.level = level
        self.mode = mode

    def _threshold_coeffs(self, coeffs, axes=None):
        coeff_array, coeff_slices, coeff_shapes = pywt.ravel_coeffs(
            coeffs, axes=axes
        )
        coeff_array = coeff_array.copy()

        keep_fraction = float(
            np.clip(1.0 / self.compression_ratio, 0.0, 1.0)
        )
        if keep_fraction <= 0.0:
            coeff_array.fill(0)
        elif keep_fraction < 1.0:
            abs_coeff = np.abs(coeff_array)
            keep_count = max(1, int(np.ceil(keep_fraction * coeff_array.size)))
            kth = coeff_array.size - keep_count
            threshold = np.partition(abs_coeff.copy(), kth)[kth]
            coeff_array[abs_coeff <= threshold] = 0

        # Quantise the kept coefficients to halve their storage footprint.
        coeff_array = coeff_array.astype(np.float16, copy=False)
        return pywt.unravel_coeffs(
            coeff_array, coeff_slices, coeff_shapes, output_format="wavedecn",
        )

    def compress_reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Return the wavelet-compressed reconstruction of ``x``."""
        x = np.asarray(x)
        axes = [ax for ax, size in enumerate(x.shape) if size > 1]
        coeffs = pywt.wavedecn(
            x, wavelet=self.wavelet, level=self.level, mode=self.mode,
            axes=axes,
        )
        coeffs = self._threshold_coeffs(coeffs, axes=axes)
        x_rec = pywt.waverecn(
            coeffs, wavelet=self.wavelet, mode=self.mode, axes=axes,
        )
        # waverecn may pad odd-sized axes; crop back to the input shape.
        x_rec = x_rec[tuple(slice(0, s) for s in x.shape)]
        return x_rec.astype(x.dtype, copy=False)


class Solver(BaseSolver):

    name = "wavelet"
    sampling_strategy = "run_once"
    requirements = ["numpy", "pip::pywavelets"]

    parameters = {
        "wavelet": ["db2"],
        "compression_ratio": [10],
        "level": [None],
        "mode": ["periodic"],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def run(self, _):
        compressor = WaveletCompressor(
            wavelet=self.wavelet,
            compression_ratio=self.compression_ratio,
            level=self.level,
            mode=self.mode,
        )
        self.fields_rec = {
            name: compressor.compress_reconstruct(arr)
            for name, arr in self.fields.items()
        }

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
