import torch
import torch.nn as nn


class FeatureRefinementModule(nn.Module):
    def __init__(self, dim: int, mode: str = "scalar", alpha_init: float = 0.0):
        super().__init__()
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if mode == "linear":
            self.proj = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.proj.weight)
        elif mode == "scalar":
            self.proj = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "scalar":
            return x * (1.0 + self.alpha)
        return x + self.alpha * self.proj(x)

