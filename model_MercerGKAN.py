
import math
import torch
import torch.nn as nn


# ==========================================================
# Normalized Hermite helper
# ==========================================================

def normalized_hermite_physicists_all(z, max_degree):
    """
    Evaluate normalized physicists' Hermite polynomials

        U_k(z) = H_k(z) / sqrt(2^k k!)

    for k = 0, ..., max_degree.

    Stable recurrence:

        U_0 = 1
        U_1 = sqrt(2) z

        U_{k+1}
        =
        sqrt(2/(k+1)) z U_k
        -
        sqrt(k/(k+1)) U_{k-1}

    No in-place tensor writes.
    """

    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")

    U_list = []

    U0 = torch.ones_like(z)
    U_list.append(U0)

    if max_degree >= 1:
        U1 = math.sqrt(2.0) * z
        U_list.append(U1)

    for k in range(1, max_degree):
        a = math.sqrt(2.0 / (k + 1.0))
        b = math.sqrt(k / (k + 1.0))

        U_next = a * z * U_list[-1] - b * U_list[-2]
        U_list.append(U_next)

    return torch.stack(U_list, dim=-1)


# ==========================================================
# Flexible Gaussian eigenbasis
# ==========================================================

class QRregEigenBasis1D(nn.Module):
    """
    One-dimensional Gaussian-kernel eigenbasis.

    Gaussian convention:

        K(x,z) = exp(-(x-z)^2 / eps^2)

    Full basis:

        phi_k(x)
        =
        sqrt(beta) exp(-delta^2 x^2) U_k(alpha beta x)

    where:

        beta = (1 + (2 / (alpha eps))^2)^(1/4)
    """

    def __init__(
        self,
        *,
        eps,
        alpha,
        num_terms,
        device="cpu",
        dtype=None,
    ):
        super().__init__()

        if eps <= 0:
            raise ValueError("eps must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if num_terms < 1:
            raise ValueError("num_terms must be >= 1")

        if dtype is None:
            dtype = torch.get_default_dtype()

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.num_terms = int(num_terms)

        eps_t = torch.tensor(float(eps), device=device, dtype=dtype)
        alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)

        beta = (1.0 + (2.0 / (alpha_t * eps_t)) ** 2) ** 0.25
        delta2 = 0.5 * alpha_t**2 * (beta**2 - 1.0)

        self.register_buffer("beta", beta)
        self.register_buffer("delta2", delta2)
        self.register_buffer("sqrt_beta", torch.sqrt(beta))

    def forward(self, x):
        """
        Input:
            x : tensor of shape (...,)

        Output:
            phi : tensor of shape (..., num_terms)
        """

        z = self.alpha * self.beta * x

        U = normalized_hermite_physicists_all(
            z,
            self.num_terms - 1,
        )

        phi = U

        envelope = torch.exp(-self.delta2 * x**2)
        phi = envelope.unsqueeze(-1) * phi

        phi = self.sqrt_beta * phi

        return phi


# ==========================================================
# Generic QR-regression layer
# ==========================================================

class QRregStage_KAN(nn.Module):
    """
    Generic QR-regression KAN layer.

    It directly learns coefficients over a truncated basis:

        y_o = sum_i sum_m coeffs[i,o,m] phi_m(x_i)

    No QR correction matrix is used.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_terms,
        eps=1.0,
        alpha=1.0,
        device="cpu",
    ):
        super().__init__()

        self.inputdim = input_dim
        self.outdim = output_dim
        self.num_terms = int(num_terms)

        self.eps = float(eps)
        self.alpha = float(alpha)

        dtype = torch.get_default_dtype()

        self.coeffs = nn.Parameter(
            torch.empty(
                input_dim,
                output_dim,
                num_terms,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.normal_(
            self.coeffs,
            mean=0.0,
            std=1.0 / (input_dim * num_terms),
        )

        self.basis = QRregEigenBasis1D(
            eps=eps,
            alpha=alpha,
            num_terms=num_terms,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        """
        Input:
            x : (batch, input_dim)

        Output:
            y : (batch, output_dim)
        """

        batch_size = x.shape[0]

        x_flat = x.reshape(-1)

        Phi = self.basis(x_flat)

        Phi = Phi.reshape(batch_size, self.inputdim, self.num_terms)

        y = torch.einsum("bid,iod->bo", Phi, self.coeffs)

        return y


# ==========================================================
# Multi-layer wrapper
# ==========================================================

class QRregEigenRawGKAN(nn.Module):
    """
    Full Gaussian eigenbasis:

        sqrt(beta) exp(-delta^2 x^2) U_k(alpha beta x)

    Gaussian convention:

        K(x,z) = exp(-(x-z)^2 / eps^2)

    This is the main eps-dependent QRreg Gaussian model.
    No column normalization.
    """

    def __init__(
        self,
        a,
        *,
        num_terms,
        eps,
        alpha=1.0,
        device="cpu",
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            QRregStage_KAN(
                input_dim=i,
                output_dim=j,
                num_terms=num_terms,
                eps=eps,
                alpha=alpha,
                device=device,
            )
            for i, j in zip(a[:-1], a[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x