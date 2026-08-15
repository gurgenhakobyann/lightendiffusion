import math
import torch


def generalized_retinex_compose(
    R: torch.Tensor,
    L: torch.Tensor,
    tau: float = 1.0,
    lam: float = 1.0,
    vartheta: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Generalized Retinex composition f(R, L) = tau * R^lambda * L^vartheta,
    from Trongtirakul, Agaian & Wu (IEEE Access, 2023), "Adaptive Single
    Low-Light Image Enhancement by Fractional Stretching in Logarithmic Domain."

    Reduces to the classical Retinex model I = R * L used in LightenDiffusion
    (Jiang et al., ECCV 2024) when tau = lambda = vartheta = 1.

    Computed in log domain for numerical stability:
        log(f + eps) = log(tau + eps) + lambda * log(R + eps) + vartheta * log(L + eps)
    """
    R_ = torch.clamp(R, min=0.0)
    L_ = torch.clamp(L, min=0.0)
    log_f = (
        math.log(tau + eps)
        + lam * torch.log(R_ + eps)
        + vartheta * torch.log(L_ + eps)
    )
    return torch.exp(log_f)


def classical_retinex_compose(R: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Baseline I = R * L, kept for classical ablation row."""
    return R * L
