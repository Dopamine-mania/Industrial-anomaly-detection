import torch
import torch.nn as nn


class Planar(nn.Module):
    """
    Planar flow from Bayes-PFL official code:
        z' = z + u * tanh(w^T z + b)
    """

    def __init__(self):
        super().__init__()
        self.h = nn.Tanh()

    def forward(self, z: torch.Tensor, u: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        # z: (B, D)
        # u: (B, D, 1)
        # w: (B, 1, D)
        # b: (B, 1, 1)
        z3 = z.unsqueeze(2)
        prod = torch.bmm(w, z3) + b
        f_z = (z3 + u * self.h(prod)).squeeze(2)

        psi = w * (1 - self.h(prod) ** 2)
        log_det_j = torch.log(torch.abs(1 + torch.bmm(psi, u)) + 1e-5).squeeze(2).squeeze(1)
        return f_z, log_det_j

