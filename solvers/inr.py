from benchopt import BaseSolver
import torch
import torch.nn as nn
from torch_inr import UniformSampler, FinerLayer, NoiseEncoding, get_coords, get_input_shape
from torch_inr.coords import get_output_dim
from tqdm import tqdm
import numpy as np


class Solver(BaseSolver):

    name = "inr"
    sampling_strategy = "run_once"
    requirements = ["pip::torch-inr@git+https://github.com/etyl/torch-inr.git", "pip::torch", "tqdm"]

    parameters = {
        "hidden_size": [128],
        "encoding_layers": [3],
        "decoding_layers": [2],
        "layer_modulation": [True],
        "lr": [1e-3],
        "epochs": [1000],
        "batch_size": [128000],
        "normalise": [False],
        "predict_dims": [[]],
        "output_regularisation": ["none"],
        "regularisation": ["none"],
        "lambda_regularisation": [1.0],
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

    def skip(self, fields: dict):
        # predict_dims must exist in every field; Tokam2D is 2D, Landau2X2V is 4D
        for d in self.predict_dims:
            for arr in fields.values():
                if d >= len(arr.shape):
                    return True, f"predict_dim {d} out of range for a {len(arr.shape)}D field"
        return False, None

    def set_objective(self, fields: dict):
        self.fields = {
            name: torch.as_tensor(arr, dtype=torch.float32)
            for name, arr in fields.items()
        }
        self.fields_min = {name: float(arr.min()) for name, arr in self.fields.items()}
        self.fields_max = {name: float(arr.max()) for name, arr in self.fields.items()}
        self.fields = {
            name: 2 * (arr - self.fields_min[name]) / (self.fields_max[name] - self.fields_min[name]) - 1
            for name, arr in self.fields.items()
        }
        self.input_shapes = {name: arr.shape for name, arr in self.fields.items()}
        self.samplers = {
            name: UniformSampler(self.fields[name], batch_size=self.batch_size, predict_dims=self.predict_dims)
            for name, arr in self.fields.items()
        }
        self.models = {
            name: nn.ModuleList([
                NoiseEncoding(len(get_input_shape(self.input_shapes[name], self.predict_dims)), self.hidden_size, n_layers=self.encoding_layers, sampler=self.samplers[name]),
                *[FinerLayer(self.hidden_size, self.hidden_size) for _ in range(self.decoding_layers-1)],
                nn.Linear(self.hidden_size, get_output_dim(self.input_shapes[name], self.predict_dims)),
            ])
            for name, arr in self.fields.items()
        }
        self.modulation_layers = None
        if self.layer_modulation:
            self.modulation_layers = {
                name: nn.ModuleList([
                    nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.encoding_layers + self.decoding_layers-1)
                ])
                for name, arr in self.fields.items()
            }
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=self.lr, eps=1e-10)
            for name, model in self.models.items()
        }

    def run(self, _):
        self.fields_rec = {}

        for name, model in self.models.items():
            model = model.to(self.device)
            if self.layer_modulation:
                modulation_layers = self.modulation_layers[name].to(self.device)
            self.samplers[name].to(self.device)
            # number of input coordinates (predicted dims are network outputs,
            # not sampled), so step count and mass scaling track the real grid
            n_points = self.samplers[name].X_target.shape[0]

            if self.predict_dims:
                x_output = torch.linspace(-1, 1, self.samplers[name].X_target.shape[1]).to(self.device)

            target_mass = self.fields[name].sum().item()
            mass = None

            grad_target = None
            if self.regularisation == "grad":
                # grad reg assumes full coord space; predict_dims path not wired in
                assert not self.predict_dims, "grad regularisation not supported with predict_dims"
                # spatial gradient of the target field, in the same [-1, 1] coord
                # units the INR sees (coord = idx/shape * 2 - 1 -> step 2/shape)
                shape = self.input_shapes[name]
                field = self.fields[name]
                # torch.gradient needs >=2 points along a dim; singleton dims
                # (e.g. landau) get a zero gradient component instead.
                dims = [i for i, s in enumerate(shape) if s >= 2]
                spacing = [2.0 / shape[i] for i in dims]
                computed = torch.gradient(field, spacing=spacing, dim=dims)
                grads = [torch.zeros_like(field) for _ in shape]
                for d, g in zip(dims, computed):
                    grads[d] = g
                grad_target = torch.stack(grads, dim=-1).reshape(-1, len(shape)).to(self.device)

            if self.batch_size >= n_points:
                total_steps = self.epochs
            else:
                total_steps = int(self.epochs * n_points / self.batch_size)

            for _ in tqdm(range(total_steps), desc=f"Training INR on {name}", mininterval=2):
                self.optimizers[name].zero_grad()
                batch = self.samplers[name].sample()
                if self.regularisation == "grad":
                    batch = batch.detach().requires_grad_(True)

                for k in range(len(model)-1):
                    batch = model[k](batch)
                    if self.layer_modulation:
                        batch = batch * modulation_layers[k](batch)
                output = model[-1](batch)

                loss = self.samplers[name].compute_loss(output)

                if self.regularisation == "mc":
                    mass = (output.sum() / batch.shape[0]) * n_points
                    loss += self.lambda_regularisation * (mass - target_mass)**2 / target_mass**2
                elif self.regularisation == "ema":
                    current_mass = (output.sum() / batch.shape[0]) * n_points
                    if mass is None:
                        mass = current_mass
                    else:
                        mass = 0.8 * mass.detach() + 0.2 * current_mass
                    loss += self.lambda_regularisation * (mass - target_mass)**2 / target_mass**2
                elif self.regularisation == "batch":
                    target_mass_batch = self.samplers[name].get_target().sum().item()
                    mass_batch = output.sum()
                    loss += self.lambda_regularisation * (mass_batch - target_mass_batch)**2 / target_mass_batch**2
                elif self.regularisation == "grad":
                    grad_inr = torch.autograd.grad(output.sum(), batch, create_graph=True)[0]
                    gt = grad_target[(self.samplers[name].idx * self.samplers[name]._multipliers).sum(dim=1)]
                    # match gradient direction only, not magnitude
                    cos = torch.nn.functional.cosine_similarity(grad_inr, gt, dim=-1)
                    loss += self.lambda_regularisation * (1 - cos).mean()

                if self.output_regularisation == "mass" and self.predict_dims:
                    target = self.samplers[name].get_target()
                    loss += self.lambda_regularisation * torch.abs(
                        output.sum(dim=1) - target.sum(dim=1)
                    ).mean() / batch.shape[0]
                if self.output_regularisation == "velocity" and self.predict_dims:
                    target = self.samplers[name].get_target()
                    loss += self.lambda_regularisation * torch.abs(
                        (x_output * output).sum(dim=1) - (x_output * target).sum(dim=1)
                    ).mean() / batch.shape[0]

                loss.backward()
                self.optimizers[name].step()

            # Reconstruct field
            shape = self.input_shapes[name]
            coords = get_coords(shape, self.predict_dims)
            output_dim = get_output_dim(shape, self.predict_dims)
            with torch.no_grad():
                field_rec = np.empty((coords.shape[0], output_dim))
                for k in range(0, coords.shape[0], self.batch_size):
                    batch = coords[k:k + self.batch_size].to(self.device)

                    for j in range(len(model)-1):
                        batch = model[j](batch)
                        if self.layer_modulation:
                            batch = batch * modulation_layers[j](batch)
                    output = model[-1](batch).cpu().numpy()

                    field_rec[k:k + self.batch_size] = output
                # invert get_target: rows are in (input_dims + predict_dims) order
                perm = [k for k in range(len(shape)) if k not in self.predict_dims] + list(self.predict_dims)
                field_rec = field_rec.reshape([shape[k] for k in perm])
                field_rec = np.transpose(field_rec, np.argsort(perm))
                field_rec = (field_rec + 1) / 2 * (self.fields_max[name] - self.fields_min[name]) + self.fields_min[name]

                if self.normalise:
                    field_rec = self.fields[name].sum().item() * field_rec / np.sum(field_rec)
            self.fields_rec[name] = field_rec

            # Free GPU memory
            model.cpu()
            self.samplers[name].to("cpu")

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
