from benchopt import BaseSolver
import torch
import torch.nn as nn
from torch_inr import UniformSampler, FinerLayer, NoiseEncoding, get_coords
from tqdm import tqdm


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
        "batch_size": [256000],
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

    def set_objective(self, fields: dict):
        self.fields = {
            name: torch.as_tensor(arr, dtype=torch.float32)
            for name, arr in fields.items()
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
            self.fields[name] = self.fields[name].to(self.device)
            for _ in tqdm(range(self.epochs), desc=f"Training INR on {name}", interval=1000):
                self.optimizers[name].zero_grad()
                batch = self.samplers[name].sample()
                output = model(batch)
                loss = self.samplers[name].compute_loss(output)
                loss.backward()
                self.optimizers[name].step()

            # Reconstruct field
            coords = get_coords(self.input_shapes[name])
            with torch.no_grad():
                self.fields_rec[name] = model(coords.to(self.device)).cpu().numpy().reshape(self.input_shapes[name])

            # Free GPU memory
            model.cpu()
            self.fields[name] = self.fields[name].cpu()

    def get_result(self) -> dict:
        return dict(fields_rec=self.fields_rec)
