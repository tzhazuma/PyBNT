"""
Image processing utilities for the PyBNT toolbox.

Includes 2D/3D slice converters, normalization, image quality
metrics (SNR, CNR, PSNR, SSIM, NRMSE), and geometric transforms.
"""

import cv2 as cv
import numpy as np
import skimage as ski  # type: ignore[import-untyped]


def to_2d(img3d: np.ndarray, z_axis: int = 2) -> list[np.ndarray]:
    """Convert a 3D image volume into a list of 2D slices.

    Args:
        img3d: 3D numpy array.
        z_axis: Axis along which to slice (0, 1, or 2).

    Returns:
        List of 2D slices.
    """
    shape = img3d.shape
    slices: list[np.ndarray] = []
    for i in range(shape[z_axis]):
        if z_axis == 0:
            slices.append(img3d[i, :, :])
        elif z_axis == 1:
            slices.append(img3d[:, i, :])
        else:
            slices.append(img3d[:, :, i])
    return slices


def to_3d(imgl: list[np.ndarray], z_axis: int = 2) -> np.ndarray:
    """Stack a list of 2D slices into a 3D volume.

    Args:
        imgl: List of 2D slices with matching shape.
        z_axis: Axis along which to stack (0, 1, or 2).

    Returns:
        3D numpy array of type ``uint8``.
    """
    shape = imgl[0].shape
    img3d = np.zeros((shape[0], shape[1], len(imgl)), dtype=np.uint8)
    for i in range(len(imgl)):
        if z_axis == 0:
            img3d[i, :, :] = imgl[i]
        elif z_axis == 1:
            img3d[:, i, :] = imgl[i]
        else:
            img3d[:, :, i] = imgl[i]
    return img3d


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize an image to the [0, 255] range using min-max scaling.

    Args:
        img: Input image array.

    Returns:
        Normalized image (same dtype as input).
    """
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)


def snr(img: np.ndarray, snr_type: str = "all") -> float:
    """Compute the signal-to-noise ratio (SNR) of an image.

    Args:
        img: 2D image array.
        snr_type: ``"all"`` uses the full image; ``"roi"`` computes SNR
                  between center ROI and a corner background patch.

    Returns:
        SNR value.
    """
    if snr_type == "all":
        return float(np.mean(img) / np.std(img))

    elif snr_type == "roi":
        x, y = img.shape
        xmin = x // 2 - x // 10
        xmax = x // 2 + x // 10
        ymin = y // 2 - y // 10
        ymax = y // 2 + y // 10
        subimg = img[xmin:xmax, ymin:ymax]

        xsurmin = 0
        xsurmax = max(1, x // 50)
        ysurmin = 0
        ysurmax = max(1, y // 50)
        surimg = img[xsurmin:xsurmax, ysurmin:ysurmax]

        return float(np.mean(subimg) / np.std(surimg))

    else:
        raise ValueError("The type is not supported")


def cnr(img: np.ndarray, cnr_type: str = "all") -> float:
    """Compute the contrast-to-noise ratio (CNR) of an image.

    Args:
        img: 2D image array.
        cnr_type: ``"all"`` uses the full image; ``"roi"`` computes CNR
                  between center ROI and a corner background patch.

    Returns:
        CNR value.
    """
    if cnr_type == "all":
        dif = np.max(img) - np.min(img)
        return float(dif / np.std(img))

    elif cnr_type == "roi":
        x, y = img.shape
        xmin = x // 2 - x // 10
        xmax = x // 2 + x // 10
        ymin = y // 2 - y // 10
        ymax = y // 2 + y // 10
        subimg = img[xmin:xmax, ymin:ymax]

        xsurmin = 0
        xsurmax = max(1, x // 50)
        ysurmin = 0
        ysurmax = max(1, y // 50)
        surimg = img[xsurmin:xsurmax, ysurmin:ysurmax]

        dif = np.mean(subimg) - np.mean(surimg)
        result = abs(dif) / np.sqrt(np.std(subimg) ** 2 + np.std(surimg) ** 2)
        return float(result)

    else:
        raise ValueError("The type is not supported")


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the Peak Signal-to-Noise Ratio between two images."""
    return float(ski.metrics.peak_signal_noise_ratio(img1, img2, data_range=255))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the Structural Similarity Index between two images."""
    return float(ski.metrics.structural_similarity(img1, img2, data_range=255))


def rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the Normalized Root Mean Square Error between two images."""
    return float(ski.metrics.normalized_root_mse(img1, img2))


def transform(
    img: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
    flip: int = 0,
) -> np.ndarray:
    """Apply a rotation, scale, and optional horizontal flip to an image.

    Args:
        img: 2D image array.
        angle: Rotation angle in degrees.
        scale: Uniform scale factor.
        flip: If 1, apply a horizontal flip after rotation.

    Returns:
        Transformed image.
    """
    rows, cols = img.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    img_out = cv.warpAffine(img, M, (cols, rows))
    if flip == 1:
        img_out = cv.flip(img_out, 1)
    return img_out


# Backward-compatible aliases with old names
To_2D = to_2d
To_3D = to_3d
