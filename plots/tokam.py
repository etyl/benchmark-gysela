from benchopt import BasePlot


class Plot(BasePlot):

    name = "Tokam Conservation"
    type = "scatter"
    options = {
        "metric": ["mass", "total_energy", "kinetic_energy", "thermal_energy"],
        "relative": [True, False],
        "dataset": ...,
        "solver": ...
    }
    requirements = ["matplotlib"]

    def plot(self, df, metric, relative, dataset, solver):
        if not dataset.startswith("Tokam2D"):
            return {}

        plots = []
        df_solver = df[df["solver_name"] == solver and df["dataset_name"] == dataset]

        if metric == "total_energy":
            metric = "energy"

        if df_solver.empty or f"objective_{metric}_gt" not in df_solver.columns:
            return {}

        energy_gt = df_solver[f"objective_{metric}_gt"].values[0]
        energy_rec = df_solver[f"objective_{metric}_rec"].values[0]
        plots.append({
            "x": list(range(len(energy_gt))),
            "y": energy_rec.tolist(),
            "label": solver,
            **self.get_style(solver)
        })
        plots.append({
            "x": list(range(len(energy_gt))),
            "y": energy_gt.tolist(),
            "label": "GT",
            **self.get_style("GT")
        })
        return plots


    def get_metadata(self, df, metric, solver):
        return dict(
            title=f"{metric} with {solver}",
            ylabel=f"{metric}",
            xlabel="Simulation steps",
        )