from benchopt import BasePlot
import re

import numpy as np
import matplotlib.pyplot as plt


class Plot(BasePlot):
    """Reference vs reconstructed 2D fields, one plot per (solver, dataset).

    The objective stores downsampled GT/reconstruction maps as
    ``objective_field_<name>_{gt,rec}``. Each field gives a GT/reconstruction
    pair, shown side by side (2 columns), colour-mapped on the GT value range so
    the reconstruction is directly comparable. Tokam shows density and
    potential; Landau shows density, momentum_x and momentum_y.
    """

    name = "Fields"
    type = "image"
    options = {
        "solver": ...,
        "dataset": ...,
    }

    @staticmethod
    def _fields(df):
        return [m.group(1) for c in df.columns
                if (m := re.fullmatch(r"objective_field_(.+)_gt", c))]

    @staticmethod
    def _colormap(gt, rec):
        # Map both onto [0, 1] using the GT range, then apply viridis -> RGB.
        gt = np.asarray(gt, dtype=float)
        rec = np.asarray(rec, dtype=float)
        lo, hi = float(gt.min()), float(gt.max())
        span = (hi - lo) or 1.0
        cmap = plt.get_cmap("viridis")
        to_rgb = lambda a: cmap(np.clip((a - lo) / span, 0, 1))[..., :3]
        return to_rgb(gt), to_rgb(rec)

    def plot(self, df, solver, dataset):
        sub = df[(df["dataset_name"] == dataset)
                 & (df["solver_name"] == solver)]
        if sub.empty:
            return []
        row = sub.iloc[0]

        images = []
        for f in self._fields(df):
            gt_col, rec_col = f"objective_field_{f}_gt", f"objective_field_{f}_rec"
            if gt_col not in row.index or rec_col not in row.index:
                continue
            gt, rec = row[gt_col], row[rec_col]
            if gt is None or rec is None:
                continue
            gt_rgb, rec_rgb = self._colormap(gt, rec)
            images.append({"image": gt_rgb, "label": f"{f} — GT"})
            images.append({"image": rec_rgb, "label": f"{f} — reconstruction"})
        return images

    def get_metadata(self, df, solver, dataset):
        return dict(
            title=f"{dataset.split('[')[0]} — {solver}",
            ncols=2,
        )
