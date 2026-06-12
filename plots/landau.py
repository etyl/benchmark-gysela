from benchopt import BasePlot


class Plot(BasePlot):

    name = "Landau Conservation"
    type = "scatter"
    options = {
        "metric": ["momentum_x", "momentum_y", "momentum_norm", "potential_energy"],
        "relative": [True, False],
        "dataset": ...,
        "solver": ...
    }

    def plot(self, df, metric, relative, dataset, solver):
        if not dataset.startswith("Landau"):
            return []

        plots = []
        df_filter = df.query(f"dataset_name == '{dataset}' and solver_name == '{solver}'")

        if metric == "total_energy":
            metric = "energy"

        if df_filter.empty or f"objective_{metric}_gt" not in df_filter.columns:
            return []

        energy_gt = df_filter[f"objective_{metric}_gt"].values[0]
        energy_rec = df_filter[f"objective_{metric}_comp"].values[0]
        plots.append({
            "x": list(range(len(energy_gt))),
            "y": energy_rec,
            "label": solver,
            **self.get_style(solver)
        })
        plots.append({
            "x": list(range(len(energy_gt))),
            "y": energy_gt,
            "label": "GT",
            **self.get_style("GT")
        })
        return plots


    def get_metadata(self, df, relative, metric, dataset, solver):
        ylabel = f"{metric} (relative)" if relative else metric
        return dict(
            title=f"{metric} with {solver}",
            ylabel=ylabel,
            xlabel="Simulation steps",
        )