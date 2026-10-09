"""
DLL Experiment 4: Variational Autoencoder (VAE)
Demonstrates probabilistic latent space encoding (mu, logvar), reparameterization trick, and generative decoding.
"""

import numpy as np
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 4):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar, z

def run_vae_experiment() -> dict:
    np.random.seed(42)
    input_data = np.random.uniform(0.1, 0.9, (1, 64)).astype(np.float32)
    tensor = torch.tensor(input_data)
    
    vae = VariationalAutoencoder(input_dim=64, latent_dim=4)
    with torch.no_grad():
        recon, mu, logvar, z = vae(tensor)
        
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = torch.sum((tensor - recon) ** 2)
    total_loss = float(recon_loss + kl_div)
    
    return {
        "experiment_name": "Variational Autoencoder (VAE)",
        "input_dim": 64,
        "latent_dim": 4,
        "mean_vector_mu": [round(float(v), 3) for v in mu.squeeze(0)],
        "log_variance_logvar": [round(float(v), 3) for v in logvar.squeeze(0)],
        "sampled_latent_z": [round(float(v), 3) for v in z.squeeze(0)],
        "kl_divergence": round(float(kl_div), 4),
        "total_vae_loss": round(total_loss, 4),
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_vae_experiment())
