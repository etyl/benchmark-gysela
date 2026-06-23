"""Self-check: _is_readable_h5 rejects corrupt cached frames."""
import h5py
import numpy as np

from benchmark_utils.landau import _is_readable_h5


def test_is_readable_h5(tmp_path):
    good = tmp_path / "good.h5"
    with h5py.File(good, "w") as h5:
        h5["fdistribu"] = np.zeros((1, 2, 2))
    assert _is_readable_h5(good)

    truncated = tmp_path / "bad.h5"
    truncated.write_bytes(b"not an hdf5 file")  # no HDF5 signature
    assert not _is_readable_h5(truncated)

    assert not _is_readable_h5(tmp_path / "missing.h5")
