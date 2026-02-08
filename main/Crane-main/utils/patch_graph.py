import math

import torch
import torch.nn.functional as F


def smooth_patch_tokens(patch_tokens: torch.Tensor, smooth_lambda: float) -> torch.Tensor:
    """
    Lightweight "graph" smoothing on ViT patch tokens via 3x3 neighborhood averaging
    on the (H,W) patch grid.

    patch_tokens: (B, L, C) where L is a perfect square.
    Returns: (B, L, C)
    """
    lam = float(smooth_lambda)
    if lam <= 0.0:
        return patch_tokens
    if patch_tokens.dim() != 3:
        return patch_tokens

    B, L, C = patch_tokens.shape
    h = int(math.isqrt(int(L)))
    if h * h != int(L):
        return patch_tokens

    x = patch_tokens.transpose(1, 2).reshape(B, C, h, h)
    y = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    out = (1.0 - lam) * x + lam * y
    return out.reshape(B, C, L).transpose(1, 2)

