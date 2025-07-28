# Core imports
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import warnings


# Setup
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("✅ All packages imported successfully!")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"💻 Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 Using device: {device}")


class SimpleVAE(nn.Module):
    """
    Simplified Variational Autoencoder for market data.

    Architecture optimized for financial time series data (stocks × features)
    Uses hyperparameters configured in the hyperparameter configuration section.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super(SimpleVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder (compress market state to latent factors)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Latent space parameters
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder (reconstruct market state from latent factors)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_log_var(h)

    def reparameterize(self, mean, log_var):
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass through VAE."""
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decode(z)
        return x_recon, mean, log_var, z

    def encode_to_latent(self, x):
        """Encode input directly to latent space (for inference)."""
        with torch.no_grad():
            mean, _ = self.encode(x)
            return mean  # Use mean for deterministic inference


def vae_loss_fn_legacy(x_recon, x, mean, log_var, beta=1.0):
    """
    ** Note - this is the standard VAE loss function.
    VAE loss function with reconstruction loss and KL divergence.

    Args:
        x_recon: Reconstructed input
        x: Original input
        mean: Latent mean
        log_var: Latent log variance
        beta: Weight for KL divergence (β-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence loss (regularization)
    kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def vae_loss_fn(x_recon, x, mean, log_var, beta=10.0, C=0.0, lambda_ortho=1e-3):
    """
    ** Note - this is the new VAE loss function with added capacity annealing and ortho penalty.
    VAE loss function with:
    - MSE reconstruction loss
    - KL capacity annealing (beta * |KL - C|)
    - Orthogonality penalty on latent mean vectors

    Args:
        x_recon: (batch, input_dim) - Reconstructed input
        x:       (batch, input_dim) - Original input
        mean:    (batch, latent_dim) - Latent mean
        log_var: (batch, latent_dim) - Latent log variance
        beta:    float - scaling factor for KL loss
        C:       float - KL capacity (target)
        lambda_ortho: float - strength of orthogonality penalty

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # KL divergence (per-sample, per-dim)
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    kl_loss = torch.mean(kl_per_sample)

    # KL annealing with capacity target C
    kl_term = beta * torch.abs(kl_loss - C)

    # Orthogonality penalty (encourages decorrelated latent dimensions)
    mu_centered = mean - mean.mean(dim=0, keepdim=True)
    cov = mu_centered.T @ mu_centered / mean.size(0)  # (latent_dim x latent_dim)
    identity = torch.eye(cov.size(0), device=mean.device)
    ortho_penalty = F.mse_loss(cov, identity)

    # Total loss
    total_loss = recon_loss + kl_term + lambda_ortho * ortho_penalty

    return total_loss, recon_loss, kl_loss, lambda_ortho * ortho_penalty
