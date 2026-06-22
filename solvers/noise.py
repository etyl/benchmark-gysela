import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):

    name = "noise"
    sampling_strategy = "run_once"
    requirements = ["numpy"]

    # Baseline "compression": additive noise scaled by the field std, to probe
    # how sensitive the downstream evaluation is to perturbations. `alpha` shapes
    # the spectrum: 0 = white, higher = more low-frequency (1=pink, 2=red/Brownian).
    parameters = {
        "noise_level": [0.01],
        "alpha": [2.0],
        "centered": [False],
    }

    def set_objective(self, fields: dict):
        self.fields = fields

    def _noise(self, shape, rng):
        """Unit-std noise with a 1/k^alpha power spectrum (low-frequency weighted)."""
        if self.alpha == 0:
            return rng.standard_normal(shape)
        spec = np.fft.fftn(rng.standard_normal(shape))
        grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
        k = np.sqrt(sum(g**2 for g in grids))
        k[(0,) * len(shape)] = 1.0  # leave DC untouched, avoid div-by-zero
        out = np.fft.ifftn(spec / k**self.alpha).real
        return out / out.std()

    def run(self, _):
        rng = np.random.default_rng()
        level = float(self.noise_level)

        self.fields_rec = {}
        for name, arr in self.fields.items():
            noise = level * np.std(arr) * self._noise(arr.shape, rng).astype(arr.dtype)
            if self.centered:
                noise -= np.mean(noise)
            self.fields_rec[name] = arr + noise

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
