import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .flows import Planar


class _InferenceBlock(nn.Module):
    def __init__(self, input_units: int, hidden_units: int, output_units: int):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, hidden_units, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_units, hidden_units, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_units, output_units, bias=True),
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


class Encoder(nn.Module):
    """
    Bayes-PFL style encoder that returns (mu, std) where std is exp(0.5*logvar).
    """

    def __init__(self, input_units: int, hidden_units: int, output_units: int):
        super().__init__()
        self.weight_mean = _InferenceBlock(input_units, hidden_units, output_units)
        self.weight_log_variance = _InferenceBlock(input_units, hidden_units, output_units)

    def forward(self, x: torch.Tensor):
        mu = self.weight_mean(x)
        logvar = self.weight_log_variance(x)
        std = torch.exp(0.5 * logvar)
        return mu, std


class Decoder(nn.Module):
    def __init__(self, input_units: int, hidden_units: int, output_units: int):
        super().__init__()
        self.weight_mean = _InferenceBlock(input_units, hidden_units, output_units)

    def forward(self, x: torch.Tensor):
        return self.weight_mean(x)


def binary_loss_function(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    z_mu: torch.Tensor,
    z_std: torch.Tensor,
    z_0: torch.Tensor,
    z_k: torch.Tensor,
    log_det_jacobians: torch.Tensor,
    z_size: int,
    beta: float = 0.0,
    if_rec: bool = True,
):
    """
    Minimal port of Bayes-PFL official `binary_loss_function`:
    - reconstruction: MSE (summed) / batch
    - KL term: E[log q(z0) - log p(zk)] - log|det|

    We ignore beta annealing and VampPrior for now (not used in our setting).
    """
    batch_size = x.size(0)
    logvar0 = torch.zeros(batch_size, z_size, device=x.device, dtype=x.dtype)

    # log p(zk) under standard Gaussian
    log_p_zk = -0.5 * torch.sum(logvar0 + z_k**2 / torch.exp(logvar0) + math.log(2 * math.pi), dim=1)
    # log q(z0) under diagonal Gaussian q(z0|x) with mean=z_mu, std=z_std
    log_q_z0 = -0.5 * torch.sum(
        (z_std.log() * 2) + (z_0 - z_mu) ** 2 / (z_std**2) + math.log(2 * math.pi),
        dim=1,
    )

    if if_rec:
        rec = F.mse_loss(x_recon, x, reduction="sum") / float(batch_size)
    else:
        rec = torch.zeros([], device=x.device, dtype=x.dtype)

    # KL (average over batch)
    kl = (log_q_z0 - log_p_zk - log_det_jacobians).mean()
    # official code returns (elbo, rec, kl); elbo isn't used
    return torch.zeros([], device=x.device, dtype=x.dtype), rec, kl


class PlanarPFL(nn.Module):
    """
    Bayes-PFL PlanarPFL (image-specific distribution).
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, embed_dim: int, num_flows: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = embed_dim
        self.num_flows = num_flows

        self.amor_u = nn.Linear(embed_dim, num_flows * embed_dim)
        self.amor_w = nn.Linear(embed_dim, num_flows * embed_dim)
        self.amor_b = nn.Linear(embed_dim, num_flows)

        self.flows = nn.ModuleList([Planar() for _ in range(num_flows)])

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        z_mu, z_std = self.encoder(x)

        u = self.amor_u(x).view(batch_size, self.num_flows, self.embed_dim, 1)
        w = self.amor_w(x).view(batch_size, self.num_flows, 1, self.embed_dim)
        b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)

        z_0 = self.reparameterize(z_mu, z_std)
        log_det_j = torch.zeros([batch_size], device=x.device, dtype=x.dtype)
        z = z_0
        for k in range(self.num_flows):
            z, inc = self.flows[k](z, u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            log_det_j = log_det_j + inc

        x_recon = self.decoder(z)
        return x_recon, z_mu, z_std, log_det_j, z_0, z


class PlanarPFLState(nn.Module):
    """
    Bayes-PFL PlanarPFL_state (image-agnostic distribution) with learnable state vector and flow params.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, embed_dim: int, num_flows: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = embed_dim
        self.num_flows = num_flows

        self.state = nn.Parameter(torch.randn(1, embed_dim))
        self.amor_u = nn.Parameter(torch.randn(1, num_flows, embed_dim, 1))
        self.amor_w = nn.Parameter(torch.randn(1, num_flows, 1, embed_dim))
        self.amor_b = nn.Parameter(torch.randn(1, num_flows, 1, 1))
        self.flows = nn.ModuleList([Planar() for _ in range(num_flows)])

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor):
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self):
        x = self.state
        z_mu, z_std = self.encoder(x)
        z_0 = self.reparameterize(z_mu, z_std)

        log_det_j = torch.zeros([x.shape[0]], device=x.device, dtype=x.dtype)
        z = z_0
        for k in range(self.num_flows):
            z, inc = self.flows[k](z, self.amor_u[:, k, :, :], self.amor_w[:, k, :, :], self.amor_b[:, k, :, :])
            log_det_j = log_det_j + inc

        x_recon = self.decoder(z)
        return x_recon, z_mu, z_std, log_det_j, z_0, z
