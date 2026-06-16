from benchopt import BasePlot
import numpy as np


class Plot(BasePlot):

    name = "AUC"
    type = "scatter"
    options = {
        "metric": ["momentum_x", "momentum_y", "momentum_norm", "potential_energy", "mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "X": ["psnr", "momentum_x", "momentum_y", "momentum_norm", "mass"],
        "dataset": ...,
    }

    def plot(self, df, metric, X, dataset):
        plots = []

        for solver in df["solver_name"].unique():
            df_filter = df[
                (df["dataset_name"] == dataset) & (df["solver_name"] == solver)
            ]

            if metric == "total_energy":
                metric = "energy"

            if df_filter.empty or f"objective_{metric}_gt" not in df_filter.columns:
                return []

            energy_gt = df_filter[f"objective_{metric}_gt"].values[0]
            energy_rec = df_filter[f"objective_{metric}_comp"].values[0]

            auc = np.sum(np.abs(np.array(energy_rec) - np.array(energy_gt)))
            if X == "psnr":
                x = df_filter["objective_psnr"].values[0]
            else:
                x = df_filter[f"objective_{X}_cons_err"].values[0]

            plots.append({
                "x": [x],
                "y": [auc],
                "label": solver,
                "marker": self.get_style(solver)["marker"],
                "color": self.get_style(solver.split('[')[0])["color"]
            })

        return plots


    def get_metadata(self, df, metric, X, dataset):
        return dict(
            title=f"{metric} AUC with {dataset}",
            ylabel="Metric AUC over time",
            xlabel=X,
        )