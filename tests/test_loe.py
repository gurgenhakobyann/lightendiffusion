import numpy as np
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.loe import compute_loe


def test_loe_identical_images():
    np.random.seed(42)
    img = np.random.rand(100, 100, 3).astype(np.float32)
    loe = compute_loe(img, img)
    print(f"Self-comparison LOE: {loe}")
    assert loe == 0.0, f"Expected 0.0 for identical images, got {loe}"
    print("test_loe_identical_images passed successfully!")


def test_loe_different_images():
    np.random.seed(42)
    img1 = np.random.rand(100, 100, 3).astype(np.float32)
    img2 = 1.0 - img1  # Inverted lightness order
    loe = compute_loe(img1, img2)
    print(f"Inverted image LOE: {loe}")
    assert loe > 0.0, f"Expected LOE > 0 for inverted images, got {loe}"
    print("test_loe_different_images passed successfully!")


if __name__ == "__main__":
    test_loe_identical_images()
    test_loe_different_images()
