import zlib

from benchopt import BaseSolver

import numpy as np
import pywt


class WaveletCompressor:
    """Transform coding: wavelet transform -> uniform scalar quantisation ->
    entropy coding. ``quant_step`` (relative to the coefficient std) sets the
    quantisation bin width: coarser steps zero out more coefficients and shrink
    the entropy-coded stream at the cost of reconstruction quality.
    """

    def __init__(self, wavelet="db2", quant_step=0.1, level=None,
                 mode="periodic"):
        self.wavelet = wavelet
        self.quant_step = float(quant_step)
        self.level = level
        self.mode = mode

    def _quantize_encode(self, coeff_array):
        """Quantise coefficients and return (dequantised coeffs, coded bytes)."""
        scale = float(np.std(coeff_array))
        if scale == 0.0:  # constant field: nothing to encode
            return coeff_array, 1
        step = self.quant_step * scale
        # int64: fine steps push indices past the int32 range; zlib strips the
        # leading zero bytes so the wider dtype barely costs anything.
        q = np.round(coeff_array / step).astype(np.int64)
        # zlib (DEFLATE = LZ77 + Huffman) entropy-codes the quantised indices;
        # the runs of zeros from coarse quantisation compress strongly.
        # ponytail: + 8 bytes for the float64 step; ignores subband-shape
        # metadata (recomputable from field shape + wavelet + level).
        n_bytes = len(zlib.compress(q.tobytes(), level=9)) + 8
        return q.astype(coeff_array.dtype) * step, n_bytes

    def compress_reconstruct(self, x: np.ndarray):
        """Wavelet reconstruction of ``x`` plus its entropy-coded size in bytes."""
        x = np.asarray(x)
        axes = [ax for ax, size in enumerate(x.shape) if size > 1]
        coeffs = pywt.wavedecn(
            x, wavelet=self.wavelet, level=self.level, mode=self.mode,
            axes=axes,
        )
        coeff_array, coeff_slices, coeff_shapes = pywt.ravel_coeffs(
            coeffs, axes=axes)
        coeff_array, n_bytes = self._quantize_encode(coeff_array)
        coeffs = pywt.unravel_coeffs(
            coeff_array, coeff_slices, coeff_shapes, output_format="wavedecn")
        x_rec = pywt.waverecn(
            coeffs, wavelet=self.wavelet, mode=self.mode, axes=axes,
        )
        # waverecn may pad odd-sized axes; crop back to the input shape.
        x_rec = x_rec[tuple(slice(0, s) for s in x.shape)]
        return x_rec.astype(x.dtype, copy=False), n_bytes


class Solver(BaseSolver):

    name = "wavelet"
    sampling_strategy = "run_once"
    requirements = ["numpy", "pip::pywavelets"]

    parameters = {
        "wavelet": ["db2"],
        "quant_step": [0.1],
        "level": [None],
        "mode": ["periodic"],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def run(self, _):
        compressor = WaveletCompressor(
            wavelet=self.wavelet,
            quant_step=self.quant_step,
            level=self.level,
            mode=self.mode,
        )
        self.fields_rec = {}
        orig_bytes = comp_bytes = 0
        for name, arr in self.fields.items():
            rec, n_bytes = compressor.compress_reconstruct(arr)
            self.fields_rec[name] = rec
            orig_bytes += arr.nbytes
            comp_bytes += n_bytes
        # Entropy coding gives a real byte size, so the ratio is bytes-based.
        self.compression_ratio_ = float(orig_bytes / max(comp_bytes, 1))

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec,
                    compression_ratio=self.compression_ratio_)
