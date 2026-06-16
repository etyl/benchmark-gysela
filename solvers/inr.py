from benchopt import BaseSolver
import torch
import torch.nn as nn
from torch_inr import UniformSampler, FinerLayer, NoiseEncoding, get_coords
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
        "lr": [1e-3],
        "epochs": [1000],
        "batch_size": [128000],
        "normalise": [False],
        "regularisation": ["none"],
        "lambda_regularisation": [1.0],
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

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
            name: UniformSampler(self.fields[name], batch_size=self.batch_size)
            for name, arr in self.fields.items()
        }
        self.models = {
            name: nn.Sequential(
                NoiseEncoding(len(self.input_shapes[name]), self.hidden_size, n_layers=self.encoding_layers, sampler=self.samplers[name]),
                *[FinerLayer(self.hidden_size, self.hidden_size) for _ in range(self.decoding_layers-1)],
                nn.Linear(self.hidden_size, 1),
            )
            for name, arr in self.fields.items()
        }
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=self.lr)
            for name, model in self.models.items()
        }

    def run(self, _):
        self.fields_rec = {}

        for name, model in self.models.items():
            model = model.to(self.device)
            self.samplers[name].to(self.device)
            n_points = self.samplers[name].X.numel()

            target_mass = self.fields[name].sum().item()
            mass = None

            if self.batch_size >= n_points:
                total_steps = self.epochs
            else:
                total_steps = int(self.epochs * n_points / self.batch_size)

            for _ in tqdm(range(total_steps), desc=f"Training INR on {name}", mininterval=2):
                self.optimizers[name].zero_grad()
                batch = self.samplers[name].sample()
                output = model(batch)
                loss = self.samplers[name].compute_loss(output)

                if self.regularisation == "mc":
                    mass = (output.sum() / batch.shape[0]) * n_points
                    loss += self.lambda_regularisation / batch.shape[0] * (mass - target_mass) ** 2
                elif self.regularisation == "ema":
                    current_mass = (output.sum() / batch.shape[0]) * n_points
                    if mass is None:
                        mass = current_mass
                    else:
                        mass = 0.9 * mass.detach() + 0.1 * current_mass
                    loss += self.lambda_regularisation / batch.shape[0] * (mass - target_mass) ** 2
                elif self.regularisation == "batch":
                    target_mass_batch = self.samplers[name].get_target().sum().item()
                    mass_batch = output.sum()
                    loss += self.lambda_regularisation / batch.shape[0] * (mass_batch - target_mass_batch) ** 2

                loss.backward()
                self.optimizers[name].step()

            # Reconstruct field
            coords = get_coords(self.input_shapes[name])
            with torch.no_grad():
                field_rec = np.empty((coords.shape[0], 1))
                for k in range(0, coords.shape[0], self.batch_size):
                    batch_coords = coords[k:k+self.batch_size]
                    output = model(batch_coords.to(self.device)).cpu().numpy()
                    field_rec[k:k+self.batch_size] = output
                field_rec =  field_rec.reshape(self.input_shapes[name])
                field_rec = (field_rec + 1) / 2 * (self.fields_max[name] - self.fields_min[name]) + self.fields_min[name]

                if self.normalise:
                    field_rec = self.fields[name].sum().item() * field_rec / np.sum(field_rec)
            self.fields_rec[name] = field_rec

            # Free GPU memory
            model.cpu()
            self.samplers[name].to("cpu")

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
