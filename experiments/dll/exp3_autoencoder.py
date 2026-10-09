"""
DLL Experiment 3: Standard Autoencoder
Demonstrates dimensionality reduction, latent space compression, and image reconstruction.
"""

import numpy as np
import torch
import torch.nn as nn

class StandardAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 8):
        super(StandardAutoencoder, self).__init__()
        # Encoder: 64 -> 32 -> 8 (latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        # Decoder: 8 -> 32 -> 64 (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def run_autoencoder_experiment() -> dict:
    np.random.seed(42)
    # 64-dim feature vector
    input_data = np.random.uniform(0.1, 0.9, (1, 64)).astype(np.float32)
    tensor = torch.tensor(input_data)
    
    model = StandardAutoencoder(input_dim=64, latent_dim=8)
    with torch.no_grad():
        latent, reconstructed = model(tensor)
        
    mse_loss = float(torch.mean((tensor - reconstructed) ** 2))
    
    return {
        "experiment_name": "Standard Autoencoder",
        "input_dimension": 64,
        "latent_dimension": 8,
        "compression_ratio": "8:1 (87.5% reduction)",
        "latent_vector_sample": [round(float(v), 3) for v in latent.squeeze(0)[:8]],
        "reconstruction_mse_loss": round(mse_loss, 5),
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_autoencoder_experiment())
