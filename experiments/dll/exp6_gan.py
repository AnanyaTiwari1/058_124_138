"""
DLL Experiment 6: Generative Adversarial Network (GAN)
Demonstrates min-max adversarial training between Generator (G) and Discriminator (D).
"""

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim: int = 16, output_dim: int = 64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def run_gan_experiment(steps: int = 5) -> dict:
    G = Generator(noise_dim=16, output_dim=64)
    D = Discriminator(input_dim=64)
    
    # Synthetic real data
    real_data = torch.randn(4, 64)
    noise = torch.randn(4, 16)
    
    # Forward passes
    fake_data = G(noise)
    d_real = D(real_data)
    d_fake = D(fake_data.detach())
    
    criterion = nn.BCELoss()
    real_labels = torch.ones(4, 1)
    fake_labels = torch.zeros(4, 1)
    
    d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
    g_loss = criterion(D(fake_data), real_labels)
    
    return {
        "experiment_name": "Generative Adversarial Network (GAN)",
        "generator_architecture": "Noise(16) -> Linear(32, LeakyReLU) -> Output(64, Tanh)",
        "discriminator_architecture": "Input(64) -> Linear(32, LeakyReLU) -> Output(1, Sigmoid)",
        "discriminator_real_score": round(float(torch.mean(d_real)), 4),
        "discriminator_fake_score": round(float(torch.mean(d_fake)), 4),
        "discriminator_loss": round(float(d_loss), 4),
        "generator_loss": round(float(g_loss), 4),
        "equilibrium_target": "D(Real) ≈ 0.5, D(Fake) ≈ 0.5",
        "status": "COMPLETED"
    }

if __name__ == "__main__":
    print(run_gan_experiment())
