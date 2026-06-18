from benchopt import BasePlot
import ast
import re

import numpy as np
import pandas as pd


class Plot(BasePlot):
    """PSNR vs a per-axis structural metric, for INR solvers only.

    Each INR run uses ``predict_dims=[k]`` (axis ``k`` is the network output).
    One point per (field, k): x = that field's metric for axis ``k``,
    y = the field's PSNR. One series per field. Runs with ``predict_dims=[]``
    (no single output axis) are skipped.
    """

    name = "PSNR vs axis metric"
    type = "scatter"
    options = {
        "metric": ["autocorr", "spectral_centroid", "total_variation", "variance"],
        "dataset": ...,
    }

    @staticmethod
    def _predict_dim(val):
        # benchopt hands plots an unpickled list, but tolerate the raw forms too:
        # a "benchopt-pkl" bytes blob (raw parquet) or the string "[0]".
        if isinstance(val, (bytes, bytearray)):
            prefix = b"\x00benchopt-pkl\x00"
            if bytes(val).startswith(prefix):
                import pickle
                val = pickle.loads(bytes(val)[len(prefix):])
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return None
        if val is None:
            return None
        val = list(val)
        return int(val[0]) if len(val) == 1 else None  # single-axis only

    @staticmethod
    def _fields(df):
        # field names from objective_<field>_psnr (the aggregate objective_psnr
        # has no <field>_ part and is excluded).
        return [m.group(1) for c in df.columns
                if (m := re.fullmatch(r"objective_(.+)_psnr", c))]

    def _series(self, df, metric, dataset):
        """{label: (xs, ys)} of (axis metric, PSNR) points across INR runs.

        One series per predicted axis (label ``axis k``, or ``<field> axis k``
        when there is more than one field), so each point's axis is legible.
        """
        if "p_solver_predict_dims" not in df.columns:
            return {}
        sub = df[(df["dataset_name"] == dataset)
                 & df["solver_name"].str.startswith("inr")]
        fields = self._fields(df)
        multi = len(fields) > 1
        series = {}
        for _, row in sub.iterrows():
            k = self._predict_dim(row["p_solver_predict_dims"])
            if k is None:
                continue
            for f in fields:
                xcol = f"objective_{f}_axis{k}_{metric}"
                ycol = f"objective_{f}_psnr"
                if xcol not in row.index or ycol not in row.index:
                    continue
                x, y = row[xcol], row[ycol]
                if pd.isna(x) or pd.isna(y):
                    continue
                label = f"{f} axis {k}" if multi else f"axis {k}"
                xs, ys = series.setdefault(label, ([], []))
                xs.append(float(x))
                ys.append(float(y))
        return series

    def plot(self, df, metric, dataset):
        plots = []
        for label, (xs, ys) in self._series(df, metric, dataset).items():
            style = self.get_style(label)
            plots.append({
                "x": xs, "y": ys, "label": label,
                "marker": style["marker"], "color": style["color"],
            })
        return plots

    def get_metadata(self, df, metric, dataset):
        # Pool every point to report an overall Pearson r in the title.
        xs, ys = [], []
        for x, y in self._series(df, metric, dataset).values():
            xs += x
            ys += y
        r = ""
        if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            r = f" (r={np.corrcoef(xs, ys)[0, 1]:.2f})"
        return dict(
            title=f"PSNR vs per-axis {metric} — INR, {dataset}{r}",
            xlabel=f"axis {metric}",
            ylabel="PSNR (dB)",
        )
