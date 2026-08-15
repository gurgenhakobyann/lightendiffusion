import torch
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.generalized_retinex import generalized_retinex_compose, classical_retinex_compose


def test_generalized_retinex_matches_classical():
    torch.manual_seed(42)
    # R and L are typically in (0, 1) from sigmoid
    R = torch.rand(4, 3, 64, 64) * 0.9 + 0.05
    L = torch.rand(4, 3, 64, 64) * 0.9 + 0.05

    classical = classical_retinex_compose(R, L)
    generalized = generalized_retinex_compose(R, L, tau=1.0, lam=1.0, vartheta=1.0, eps=1e-4)

    # Floating-point tolerance with small eps offset
    diff = torch.abs(classical - generalized)
    max_diff = torch.max(diff).item()
    print(f"Max absolute difference between classical and generalized (tau=1, lam=1, vartheta=1): {max_diff:.6f}")
    assert max_diff < 1e-3, f"Difference too large: {max_diff}"
    print("test_generalized_retinex_matches_classical passed successfully!")


def test_generalized_retinex_scaling():
    torch.manual_seed(42)
    R = torch.full((1, 1, 4, 4), 0.5)
    L = torch.full((1, 1, 4, 4), 0.5)

    # f = 2.0 * (0.5)^2 * (0.5)^1 = 2.0 * 0.25 * 0.5 = 0.25
    composed = generalized_retinex_compose(R, L, tau=2.0, lam=2.0, vartheta=1.0, eps=1e-6)
    expected = 0.25
    diff = torch.abs(composed - expected).max().item()
    assert diff < 1e-3, f"Expected ~0.25, got diff {diff}"
    print("test_generalized_retinex_scaling passed successfully!")


if __name__ == "__main__":
    test_generalized_retinex_matches_classical()
    test_generalized_retinex_scaling()
