import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    import PIL.Image as Image
    _HAS_CV2 = False


def compute_loe(I_ref: np.ndarray, I_enh: np.ndarray, downsample_size: int = 50) -> float:
    """
    Lightness Order Error (Wang et al., "Naturalness Preserved Enhancement
    Algorithm for Non-Uniform Illumination Images," IEEE TIP 2013), used as
    the calibration metric in the Agaian et al. paper (Eq. 14-15, Fig. 3).

    Lower is better. Both images must be same size, uint8 or float [0,1] RGB.
    Downsamples to `downsample_size` x `downsample_size` for tractable O(N^2)
    pairwise comparison, matching common practice in the LLIE literature.
    """
    def luminance(img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        # If shape is (H, W, C) or (C, H, W)
        if img.ndim == 3 and img.shape[2] in [1, 3]:
            L = np.max(img, axis=2)
        elif img.ndim == 3 and img.shape[0] in [1, 3]:
            L = np.max(img, axis=0)
        else:
            L = img
        
        if _HAS_CV2:
            L_small = cv2.resize(L, (downsample_size, downsample_size), interpolation=cv2.INTER_AREA)
        else:
            img_pil = Image.fromarray((L * 255).astype(np.uint8))
            img_pil = img_pil.resize((downsample_size, downsample_size), Image.BILINEAR)
            L_small = np.array(img_pil, dtype=np.float32) / 255.0

        return L_small.flatten()

    Lr = luminance(I_ref)
    Le = luminance(I_enh)
    n = len(Lr)

    # Pairwise relative-order comparison, vectorized:
    # Ur[i, j] = 1 if Lr[i] >= Lr[j] else 0
    Ur = (Lr[:, None] >= Lr[None, :]).astype(np.uint8)
    Ue = (Le[:, None] >= Le[None, :]).astype(np.uint8)
    RD = np.sum(np.abs(Ur - Ue), axis=1)
    loe = np.mean(RD) / n
    return float(loe)
