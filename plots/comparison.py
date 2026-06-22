from benchopt import BasePlot
import numpy as np


class Plot(BasePlot):

    name = "Conservation Comparison"
    type = "scatter"
    options = {
        "Y": ["momentum_x", "momentum_y", "momentum_norm", "potential_energy", "mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "X": ["psnr", "momentum_x", "momentum_y", "momentum_norm", "mass"],
        "dataset": ...,
    }

    def plot(self, df, Y, X, dataset):
        plots = []

        for solver in df["solver_name"].unique():
            df_filter = df[
                (df["dataset_name"] == dataset) & (df["solver_name"] == solver)
            ]

            if Y == "total_energy":
                Y = "energy"

            if df_filter.empty or f"objective_{Y}_gt" not in df_filter.columns:
                return []

            energy_gt = df_filter[f"objective_{Y}_gt"].values[0]
            energy_rec = df_filter[f"objective_{Y}_comp"].values[0]
            if not isinstance(energy_gt, list) or not isinstance(energy_rec, list):
                continue

            # the compressed restart can be shorter than the reference (e.g. it
            # diverged early); compare over the overlap, like trajectory_diff.
            gt = np.asarray(energy_gt, dtype=float)
            rec = np.asarray(energy_rec, dtype=float)
            n = min(gt.size, rec.size)
            l1 = float(np.sum(np.abs(rec[:n] - gt[:n])))
            if X == "psnr":
                x = df_filter["objective_psnr"].values[0]
            else:
                x = df_filter[f"objective_{X}_cons_err"].values[0]

            plots.append({
                "x": [x],
                "y": [l1],
                "label": solver,
                "marker": self.get_style(solver)["marker"],
                "color": self.get_style(solver.split('[')[0])["color"]
            })

        return plots


    def get_metadata(self, df, Y, X, dataset):
        return dict(
            title=f"{Y} L1 loss with {dataset}",
            ylabel="Metric L1 loss over time",
            xlabel=X,
        )