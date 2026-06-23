import numpy as np
from benchopt import BaseSolver
import torch
import torch.nn as nn
import tqdm


class AECompressor:
    def __init__(self, data_shape, n_feature_axes, latent_size, lr=1e-3, n_steps=100, device="cpu", batch_size=256):
        self.n_feature_axes = n_feature_axes
        self.lr = float(lr)
        self.device = device
        self.n_steps = n_steps
        self.batch_size = batch_size
        split = max(1, len(data_shape) - self.n_feature_axes)
        self.n_samples = int(np.prod(data_shape[:split]))
        self.features = int(np.prod(data_shape[split:]))
        self.encoder = nn.Sequential(
            nn.Linear(self.features, self.features // 2),
            nn.ReLU(),
            nn.Linear(self.features // 2, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, self.features),
        )
        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            eps=1e-10
        )

    def _array_to_matrix(self, f):
        """Flatten an nD field into a 2D (samples, features) matrix."""
        return f.reshape(self.n_samples, self.features)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def compress_reconstruct(self, x: np.ndarray):
        X = self._array_to_matrix(x.astype(np.float32, copy=False))
        X_tensor = torch.from_numpy(X).to(self.device)

        mean = X_tensor.mean()
        std = X_tensor.std()
        X_tensor = (X_tensor - mean) / (std + 1e-10)
        self.to(self.device)

        criterion = nn.MSELoss()
        progress_bar = tqdm.tqdm(range(self.n_steps), desc="Training Autoencoder")
        perm = torch.randperm(self.n_samples, device=self.device)
        pos = 0
        for step in progress_bar:
            if pos + self.batch_size > self.n_samples:
                perm = torch.randperm(self.n_samples, device=self.device)
                pos = 0
            X_batch = X_tensor[perm[pos:pos + self.batch_size]]
            pos += self.batch_size
            self.optim.zero_grad()
            latent = self.encoder(X_batch)
            X_reconstructed = self.decoder(latent)
            loss = criterion(X_reconstructed, X_batch)
            loss.backward()
            self.optim.step()
            if step % 10 == 0 or step == self.n_steps - 1:
                progress_bar.set_postfix({"loss": loss.item()})

        # After training, reconstruct the input.
        with torch.no_grad():
            latent = self.encoder(X_tensor)
            X_reconstructed = self.decoder(latent)
        # Denormalize back to original range.
        X_reconstructed = X_reconstructed * std + mean

        self.to("cpu")  # Move model back to CPU after training
        return X_reconstructed.cpu().numpy().reshape(x.shape)


class Solver(BaseSolver):

    name = "autoencoder"
    sampling_strategy = "run_once"
    requirements = ["pip::torch", "pip::tqdm"]

    parameters = {
        "n_feature_axes": [2],
        "latent_size": [5],
        "batch_size": [256],
        "n_steps": [100],
        "lr": [1e-3],
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
    }

    def set_objective(self, fields: dict):
        self.fields = fields
        self.compressors = {
            name: AECompressor(
                data_shape=arr.shape,
                n_feature_axes=self.n_feature_axes,
                latent_size=self.latent_size,
                batch_size=self.batch_size,
                lr=self.lr,
                n_steps=self.n_steps,
                device=self.device
            )
            for name, arr in fields.items()
        }

    def run(self, _):
        self.fields_rec = {}
        for name in self.fields:
            arr = self.fields[name]
            compressor = self.compressors[name]
            rec = compressor.compress_reconstruct(arr)
            self.fields_rec[name] = rec

    def get_result(self) -> dict:
        field_name = next(iter(self.fields))
        original_size = np.prod(self.fields[field_name].shape)

        # Calculate the size of the reconstructed representation:
        rec_size = 0
        rec_size += self.compressors[field_name].n_samples * self.latent_size  # latent representation
        rec_size += 2 # the min and max values for normalization
        for layer in self.compressors[field_name].decoder: # the decoder weights
            if isinstance(layer, nn.Linear):
                rec_size += layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)

        compression_ratio = original_size / rec_size

        return dict(fields_rec=self.fields_rec, compression_ratio=compression_ratio)
