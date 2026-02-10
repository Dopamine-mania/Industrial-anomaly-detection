import torch
import torch.nn as nn


class ResCLIP(nn.Module):
    """
    ResCLIP-style residual fusion module (training-free).

    out = (1 - alpha) * base + alpha * residual
    alpha can be a fixed scalar or a learnable parameter (optional).
    """

    def __init__(self, alpha: float = 0.5, learnable_alpha: bool = False):
        super().__init__()
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)), persistent=False)
        self.learnable_alpha = bool(learnable_alpha)

    def forward(self, base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        if self.learnable_alpha:
            a = torch.clamp(a, 0.0, 1.0)
        return (1.0 - a) * base + a * residual


class ResCLIPResidualAttention(ResCLIP):
    """
    Alias class name to match "Residual Attention" wording in the client request.
    """

    pass

