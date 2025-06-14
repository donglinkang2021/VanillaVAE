# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.fc = nn.Linear(28 * 28, 512)
        self.mu = nn.Linear(512, latent_dims)
        self.log_var = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
class AutoEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z = self.encode(input)
        x_recon = self.decode(z)
        loss = F.mse_loss(x_recon, input)
        return loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, latent_dims):
        super(VariationalAutoEncoder, self).__init__(latent_dims)
        self.encoder = VariationalEncoder(latent_dims)
        self.kl_ratio = latent_dims / (28 * 28) # IMPORTANT: Adjust KL divergence loss to each pixel

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        kl_loss = 0.5 * (mu**2 + log_var.exp() - log_var - 1).mean()
        recon_loss = F.mse_loss(x_recon, input)
        loss = recon_loss + kl_loss * self.kl_ratio
        return loss
