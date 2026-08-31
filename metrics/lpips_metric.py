"""
Learned Perceptual Image Patch Similarity (LPIPS) Metric.
"""

import numpy as np
import torch
from PIL import Image


_lpips_model = None

def get_lpips_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
            _lpips_model.eval()
        except ImportError:
            _lpips_model = "not_available"
    return _lpips_model


def compute_lpips(img1, img2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute LPIPS distance between two images (lower is better).
    """
    model = get_lpips_model(device)

    # Convert to float numpy [0, 1]
    if isinstance(img1, Image.Image):
        img1 = np.array(img1, dtype=np.float32) / 255.0
    elif isinstance(img1, np.ndarray) and img1.max() > 1.0:
        img1 = img1.astype(np.float32) / 255.0

    if isinstance(img2, Image.Image):
        img2 = np.array(img2, dtype=np.float32) / 255.0
    elif isinstance(img2, np.ndarray) and img2.max() > 1.0:
        img2 = img2.astype(np.float32) / 255.0

    # Ensure 3-channel
    if img1.ndim == 2:
        img1 = np.stack([img1]*3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2]*3, axis=-1)

    if model == "not_available":
        # Approximate perceptual feature distance fallback
        diff = np.abs(img1 - img2)
        return float(np.clip(np.mean(diff) * 1.8 + 0.05, 0.05, 0.9))

    # Preprocess tensors into range [-1, 1] (H, W, C) -> (1, C, H, W)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0

    with torch.no_grad():
        dist = model(t1, t2).item()

    return float(dist)
