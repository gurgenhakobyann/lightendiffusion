import cv2
import numpy as np
import torch
import torch.nn.functional as F


def simplest_color_balance(img, low_clip=0.01, high_clip=0.99):
    """
    Simplest Color Balance (histogram percentile stretching).
    Works on 2D or 3D numpy arrays in range [0, 1].
    """
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        channel = img[:, :, i]
        low_val = np.percentile(channel, low_clip * 100)
        high_val = np.percentile(channel, high_clip * 100)
        if high_val > low_val:
            channel = np.clip((channel - low_val) / (high_val - low_val), 0.0, 1.0)
        img[:, :, i] = channel
    return img


def multi_scale_retinex(img, sigmas=[15, 80, 250], weights=[1/3, 1/3, 1/3]):
    """
    Multi-Scale Retinex (MSR) on [0, 1] float image.
    """
    retinex = np.zeros_like(img, dtype=np.float32)
    img_float = np.clip(img, 1e-6, 1.0)
    for sigma, weight in zip(sigmas, weights):
        # Gaussian blur per channel
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        blur = np.clip(blur, 1e-6, 1.0)
        retinex += weight * (np.log(img_float) - np.log(blur))
    return retinex


def msrcr(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0, low_clip=0.01, high_clip=0.99):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR) - Jobson et al. (IEEE TIP 1997).
    Applies color restoration factor C_i and canonical histogram stretching.
    """
    img_float = np.array(img, dtype=np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    # 1. Multi-scale Retinex
    msr = multi_scale_retinex(img_float, sigmas=sigmas)

    # 2. Color Restoration Factor C_i = beta * [log(alpha * I_i) - log(sum(I_c))]
    sum_channels = np.sum(img_float, axis=2, keepdims=True) + 1e-6
    crf = beta * (np.log(alpha * img_float + 1e-6) - np.log(sum_channels))

    # 3. Combine
    msrcr_img = crf * msr

    # 4. Canonical dynamic range stretching
    msrcr_img = simplest_color_balance(msrcr_img, low_clip=low_clip, high_clip=high_clip)
    return np.clip(msrcr_img, 0.0, 1.0)


def msrcp(img, sigmas=[15, 80, 250], low_clip=0.01, high_clip=0.99):
    """
    Multi-Scale Retinex with Chromaticity Preservation (MSRCP) - Petro et al. (IPOL 2014).
    Applies MSR on Intensity channel and scales RGB to preserve original chromaticity.
    """
    img_float = np.array(img, dtype=np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    # 1. Intensity channel
    intensity = np.mean(img_float, axis=2, keepdims=True)

    # 2. MSR on intensity
    msr_int = multi_scale_retinex(intensity, sigmas=sigmas)
    msr_int = simplest_color_balance(msr_int, low_clip=low_clip, high_clip=high_clip)

    # 3. Project back to RGB preserving chromaticity
    ratio = (msr_int + 1e-6) / (intensity + 1e-6)
    out = img_float * ratio
    return np.clip(out, 0.0, 1.0)


def guided_chroma_correction(enhanced_img, low_light_img, alpha=0.5):
    """
    Guided Chrominance Alignment:
    Transfers the true natural chromaticity from the low-light input onto
    the brightened luminance of the diffusion-generated image.
    alpha: blend factor between generated color (0.0) and guided color (1.0).
    """
    enh = np.array(enhanced_img, dtype=np.float32)
    low = np.array(low_light_img, dtype=np.float32)
    if enh.max() > 1.0: enh /= 255.0
    if low.max() > 1.0: low /= 255.0

    # Luminance of generated image
    Y_enh = 0.299 * enh[:, :, 0] + 0.587 * enh[:, :, 1] + 0.114 * enh[:, :, 2]
    Y_enh = np.clip(Y_enh[:, :, None], 1e-6, 1.0)

    # Chrominance ratios of low-light image
    sum_low = np.sum(low, axis=2, keepdims=True) + 1e-6
    chroma_low = low / sum_low

    # Chrominance ratios of enhanced image
    sum_enh = np.sum(enh, axis=2, keepdims=True) + 1e-6
    chroma_enh = enh / sum_enh

    # Blended chromaticity
    chroma_target = (1.0 - alpha) * chroma_enh + alpha * chroma_low

    # Recombine luminance and chromaticity
    corrected = Y_enh * (chroma_target / (np.mean(chroma_target, axis=2, keepdims=True) + 1e-6))
    return np.clip(corrected, 0.0, 1.0)


def apply_color_correction(img, low_img=None, method="msrcp"):
    """
    Unified entrypoint for color correction algorithms.
    method: 'msrcr', 'msrcp', 'guided', 'none'
    """
    if method is None or method.lower() == "none":
        return img
    elif method.lower() == "msrcr":
        return msrcr(img)
    elif method.lower() == "msrcp":
        return msrcp(img)
    elif method.lower() == "guided":
        if low_img is not None:
            return guided_chroma_correction(img, low_img)
        return msrcp(img)
    else:
        raise ValueError(f"Unknown color correction method: {method}")
