from benchopt import BasePlot


class Plot(BasePlot):

    name = "Conservation curves"
    type = "scatter"
    options = {
        "metric": ["momentum_x", "momentum_y", "momentum_norm", "potential_energy", "mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "dataset": ...,
    }

    def plot(self, df, metric, dataset):
        plots = []
        df_filter = df[df["dataset_name"] == dataset]

        if metric == "total_energy":
            metric = "energy"

        if df_filter.empty or f"objective_{metric}_gt" not in df_filter.columns:
            return []

        energy_gt = df_filter[f"objective_{metric}_gt"].values[0]
        if not isinstance(energy_gt, list):
            return []
        plots.append({
            "x": list(range(len(energy_gt))),
            "y": energy_gt,
            "label": "GT",
            **self.get_style("GT")
        })

        for solver in df_filter["solver_name"].unique():
            energy_rec = df_filter[
                df_filter["solver_name"] == solver
            ][f"objective_{metric}_comp"].values[0]
            if not isinstance(energy_rec, list):
                continue
            plots.append({
                "x": list(range(len(energy_rec))),
                "y": energy_rec,
                "label": solver,
                "marker": self.get_style(solver)["marker"],
                "color": self.get_style(solver.split('[')[0])["color"]
            })
        return plots

    def get_metadata(self, df, metric, dataset):
        return dict(
            title=f"{metric}",
            ylabel=metric,
            xlabel="Simulation steps",
        )