"""Per-axis structural metrics of a field.

Used to correlate the structure of an axis with how well an INR reconstructs
when that axis is its output dimension (``predict_dims=[k]``). All metrics are
computed on the 1D slices ("lines") along the axis and averaged over the lines.
"""

import numpy as np

EPS = 1e-12


def _lines(arr, axis):
    """Reshape ``arr`` into (n_lines, axis_len): each row a 1D slice along axis."""
    a = np.moveaxis(np.asarray(arr, dtype=np.float64), axis, -1)
    return a.reshape(-1, a.shape[-1])


def axis_metrics(arr, axis):
    """Structural metrics for one axis of a field, averaged over all lines.

    - ``autocorr``: mean lag-1 autocorrelation (1 = smooth/predictable along
      the axis, ~0 = uncorrelated).
    - ``spectral_centroid``: power-weighted mean frequency in [0, 1]
      (0 = low-frequency/smooth, 1 = Nyquist/rapidly varying).
    - ``total_variation``: mean |first difference| per step, in global-std units.
    - ``variance``: variance along the axis as a fraction of total field variance.
    """
    n = int(arr.shape[axis])
    nan = {"autocorr": np.nan, "spectral_centroid": np.nan,
           "total_variation": np.nan, "variance": np.nan}
    if n < 2:
        return nan
    lines = _lines(arr, axis)
    std = lines.std()

    # lag-1 autocorrelation, averaged over non-constant lines
    a, b = lines[:, :-1], lines[:, 1:]
    am = a - a.mean(1, keepdims=True)
    bm = b - b.mean(1, keepdims=True)
    den = np.sqrt((am**2).sum(1) * (bm**2).sum(1))
    ok = den > EPS
    ac = (am * bm).sum(1)[ok] / den[ok]
    autocorr = float(ac.mean()) if ac.size else np.nan

    # mean total variation per step, normalised by the field's std
    total_variation = float(np.abs(np.diff(lines, axis=1)).mean() / max(std, EPS))

    # spectral centroid in [0, 1] (drop DC by removing each line's mean)
    power = np.abs(np.fft.rfft(lines - lines.mean(1, keepdims=True), axis=1)) ** 2
    p = power.sum(0)
    freq = np.fft.rfftfreq(n)  # 0 .. 0.5
    spectral_centroid = float((freq * p).sum() / max(p.sum(), EPS) / 0.5)

    # variance along the axis as a fraction of total variance
    variance = float(lines.var(1).mean() / max(float(np.asarray(arr).var()), EPS))

    return {"autocorr": autocorr, "spectral_centroid": spectral_centroid,
            "total_variation": total_variation, "variance": variance}


def field_axis_metrics(fields):
    """Flat dict of per-axis metrics for every field.

    Keys are ``<field>_axis{k}_<metric>`` so they survive as flat parquet
    columns (prefixed ``objective_`` by benchopt).
    """
    out = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        for k in range(arr.ndim):
            for m, v in axis_metrics(arr, k).items():
                out[f"{name}_axis{k}_{m}"] = v
    return out


if __name__ == "__main__":
    # A smooth sinusoid along axis 0 must look more correlated, lower-frequency
    # and lower-variation than white noise along the same axis.
    rng = np.random.default_rng(0)
    n = 256
    smooth = np.sin(np.linspace(0, 6 * np.pi, n))[:, None] + np.zeros((1, 8))
    noisy = rng.standard_normal((n, 8))
    ms, mn = axis_metrics(smooth, 0), axis_metrics(noisy, 0)
    assert ms["autocorr"] > mn["autocorr"], (ms, mn)
    assert ms["spectral_centroid"] < mn["spectral_centroid"], (ms, mn)
    assert ms["total_variation"] < mn["total_variation"], (ms, mn)
    assert set(field_axis_metrics({"f": smooth})) == {
        f"f_axis{k}_{m}" for k in (0, 1)
        for m in ("autocorr", "spectral_centroid", "total_variation", "variance")
    }
    print("axis_metrics self-check passed")
    print("smooth:", ms)
    print("noisy: ", mn)
