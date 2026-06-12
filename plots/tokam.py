from benchopt import BasePlot


class Plot(BasePlot):

    name = "Tokam Conservation"
    type = "scatter"
    options = {
        "metric": ["mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "dataset": ...,
        "solver": ...
    }

    def plot(self, df, metric, dataset, solver):
        if not dataset.startswith("Tokam2D"):
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


    def get_metadata(self, df, metric, dataset, solver):
        return dict(
            title=f"{metric} with {solver}",
            ylabel=metric,
            xlabel="Simulation steps",
        )