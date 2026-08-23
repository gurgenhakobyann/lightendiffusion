"""
Lightness Order Error (LOE) Metric Implementation.

Based on: Wang et al., "Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images", IEEE TIP 2013.
"""

import numpy as np
from PIL import Image
import torch


def compute_loe(img1, img2, downsample_size=(100, 100)):
    """
    Computes the Lightness Order Error (LOE) between two images.
    
    Args:
        img1: First image, numpy array (H, W, 3) or (H, W), float in [0, 1] or uint8 in [0, 255]
        img2: Second image, numpy array (H, W, 3) or (H, W), float in [0, 1] or uint8 in [0, 255]
        downsample_size: Tuple (w, h) to downsample for fast calculation (standard is 100x100).
    
    Returns:
        float: LOE value (lower is better, 0 is perfect lightness preservation).
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Convert to float32 [0, 1]
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    else:
        img1 = img1.astype(np.float32)
        if img1.max() > 1.0:
            img1 = img1 / 255.0

    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0
    else:
        img2 = img2.astype(np.float32)
        if img2.max() > 1.0:
            img2 = img2 / 255.0

    # Ensure lightness map L(x, y) = max_{c \in {R, G, B}} I_c(x, y)
    if img1.ndim == 3 and img1.shape[2] == 3:
        L1 = np.max(img1, axis=2)
    elif img1.ndim == 3 and img1.shape[0] == 3:
        L1 = np.max(img1, axis=0)
    else:
        L1 = img1

    if img2.ndim == 3 and img2.shape[2] == 3:
        L2 = np.max(img2, axis=2)
    elif img2.ndim == 3 and img2.shape[0] == 3:
        L2 = np.max(img2, axis=0)
    else:
        L2 = img2

    # Downsample for computational efficiency (Wang et al. standard is 100x100 or 50x50) using PIL
    if downsample_size is not None:
        w, h = downsample_size
        L1_pil = Image.fromarray((L1 * 255.0).astype(np.uint8))
        L2_pil = Image.fromarray((L2 * 255.0).astype(np.uint8))
        L1 = np.array(L1_pil.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
        L2 = np.array(L2_pil.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0

    # Flatten
    l1_flat = L1.flatten()
    l2_flat = L2.flatten()
    N = len(l1_flat)

    # Vectorized pairwise comparison
    # diff_1[i, j] = 1 if l1_flat[i] >= l1_flat[j] else 0
    diff_1 = (l1_flat[:, None] >= l1_flat[None, :]).astype(np.int32)
    diff_2 = (l2_flat[:, None] >= l2_flat[None, :]).astype(np.int32)

    # XOR / mismatch indicator
    mismatch = np.bitwise_xor(diff_1, diff_2)
    loe = np.sum(mismatch) / float(N)
    return float(loe)
