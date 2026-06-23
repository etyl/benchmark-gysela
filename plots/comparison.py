from benchopt import BasePlot
import numpy as np


class Plot(BasePlot):

    name = "Comparison"
    type = "scatter"
    options = {
        "Y": ["psnr", "compression_ratio", "momentum_x", "momentum_y", "momentum_norm", "potential_energy", "mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "X": ["psnr", "compression_ratio", "momentum_x", "momentum_y", "momentum_norm", "mass"],
        "dataset": ...,
    }

    def _value(self, df_filter, name, role):
        # psnr / compression_ratio are plain per-run scalars (either axis).
        # Conservation quantities are summarised as a static cons_err on X and a
        # trajectory L1 loss on Y. Returns None when the column is unavailable.
        if name in ("psnr", "compression_ratio"):
            return float(df_filter[f"objective_{name}"].values[0])
        if role == "x":
            col = f"objective_{name}_cons_err"
            return float(df_filter[col].values[0]) if col in df_filter.columns else None

        # role == "y": L1 loss between reference and reconstructed trajectories.
        name = "energy" if name == "total_energy" else name
        gt_col, rec_col = f"objective_{name}_gt", f"objective_{name}_comp"
        if gt_col not in df_filter.columns:
            return None
        gt, rec = df_filter[gt_col].values[0], df_filter[rec_col].values[0]
        if not isinstance(gt, list) or not isinstance(rec, list):
            return None
        # the compressed restart can be shorter than the reference (e.g. it
        # diverged early); compare over the overlap, like trajectory_diff.
        gt = np.asarray(gt, dtype=float)
        rec = np.asarray(rec, dtype=float)
        n = min(gt.size, rec.size)
        return float(np.sum(np.abs(rec[:n] - gt[:n])))

    def plot(self, df, Y, X, dataset):
        plots = []

        for solver in df["solver_name"].unique():
            df_filter = df[
                (df["dataset_name"] == dataset) & (df["solver_name"] == solver)
            ]
            if df_filter.empty:
                continue

            y = self._value(df_filter, Y, "y")
            x = self._value(df_filter, X, "x")
            if x is None or y is None:
                continue

            plots.append({
                "x": [x],
                "y": [y],
                "label": solver,
                "marker": self.get_style(solver)["marker"],
                "color": self.get_style(solver.split('[')[0])["color"]
            })

        return plots


    def get_metadata(self, df, Y, X, dataset):
        dataset_name = dataset.split('[')[0]
        return dict(
            title=f"{Y} vs {X} on {dataset_name}",
            ylabel=Y,
            xlabel=X,
        )
