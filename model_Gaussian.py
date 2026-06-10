# ==========================================================
# Gaussian KAN (Multi-layer, grid-based)
# Kernel: exp(-r^2 / eps^2)
# ==========================================================

import torch
import torch.nn as nn


class Gaussian_KAN(nn.Module):
    """
    Gaussian kernel layer with grid-distributed centers:

        exp(-(x - c_j)^2 / eps^2)

    eps is a length-scale.
    """

    def __init__(self, input_dim, output_dim, num_grid, eps=1.0, device="cpu"):
        super().__init__()

        self.inputdim = input_dim
        self.outdim   = output_dim
        self.num_grid = num_grid
        self.eps      = eps

        self.coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, num_grid, device=device)
        )

        nn.init.normal_(
            self.coeffs,
            mean=0.0,
            std=1.0 / (input_dim * num_grid),
        )

        self.register_buffer(
            "centers",
            torch.linspace(0.0, 1.0, num_grid, device=device),
        )

    def forward(self, x):
        x = x.view(-1, self.inputdim, 1).expand(-1, -1, self.num_grid)

        r2 = (x - self.centers) ** 2

        phi = torch.exp(-r2 / (self.eps ** 2))

        y = torch.einsum("bid,iod->bo", phi, self.coeffs)

        return y


class GKAN(nn.Module):
    """
    Multi-layer Gaussian KAN (grid-based)
    """

    def __init__(
        self,
        a,
        *,
        num_grid,
        eps,
        device="cpu",
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            Gaussian_KAN(
                i,
                j,
                num_grid=num_grid,
                eps=eps,
                device=device,
            )
            for i, j in zip(a[:-1], a[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x