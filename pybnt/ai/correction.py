"""Image correction and denoising API.

Provides correction functions for 2D and 3D images with fallback
classical algorithms when no trained deep-learning model is available.
"""

import logging
import os
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)


def correct_image(
    image: Union[np.ndarray, str],
    method: str = "denoise",
    **kwargs: Any,
) -> np.ndarray:
    """Correct an image using the specified method.

    Supports both 2D and 3D inputs.  When ``image`` is a string it is
    treated as a file path and loaded automatically.

    Parameters
    ----------
    image : np.ndarray or str
        Input image as a NumPy array (2D or 3D) or a path to an image file.
    method : str, optional
        Correction method to apply.  Supported values:

        - ``"denoise"`` – noise removal (default)
        - ``"motion"`` – motion artifact correction via averaging

    **kwargs
        Additional keyword arguments forwarded to the underlying algorithm.

    Returns
    -------
    np.ndarray
        Corrected image with the same shape and dtype as the input.

    Raises
    ------
    ValueError
        If the ``method`` is not recognised.
    TypeError
        If ``image`` is neither a string nor a NumPy array.

    Examples
    --------
    >>> import numpy as np
    >>> from pybnt.ai.correction import correct_image
    >>> img = np.random.rand(64, 64).astype(np.uint8)
    >>> result = correct_image(img, method="denoise")
    >>> result.shape
    (64, 64)
    """
    if isinstance(image, str):
        image = _load_image(image)

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a NumPy array or file path, got {type(image).__name__}"
        )

    if method == "denoise":
        return _denoise(image, **kwargs)
    elif method == "motion":
        return _correct_motion(image, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Supported: 'denoise', 'motion'"
        )


def train_correction_model(
    data_dir: str,
    save_dir: str,
    **kwargs: Any,
) -> str:
    """Train an image-correction model and save it to disk.

    This is a placeholder that sets up the training pipeline.  The
    actual deep-learning training logic (dataset loading, model
    definition, optimisation loop, checkpointing) is delegated to the
    ``AI_correct/`` directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing training images.
    save_dir : str
        Directory where the trained model and artefacts will be saved.
    **kwargs
        Additional configuration forwarded to the training runner
        (e.g. ``epochs``, ``batch_size``, ``lr``).

    Returns
    -------
    str
        Path to the saved model checkpoint.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or is empty.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        "Training correction model from %s -> %s (kwargs: %s)",
        data_dir,
        save_dir,
        kwargs,
    )

    # Placeholder: real training goes here.  For now we just return
    # the save directory so callers can know where the output will be.
    model_path = os.path.join(save_dir, "correction_model.pth")
    model_path = os.path.abspath(model_path)

    logger.info("Model will be saved to %s", model_path)
    return model_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk into a NumPy array.

    Returns a 2-D (grayscale) or 3-D (H, W, C) array.
    """
    try:
        import cv2
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return image
    except ImportError:
        pass

    try:
        from skimage import io
        return io.imread(path)
    except ImportError:
        pass

    raise ImportError(
        "Neither OpenCV (cv2) nor scikit-image is installed. "
        "Install one to load images from disk."
    )


def _denoise(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Apply denoising to ``image``.

    Uses classical algorithms as a fallback when no trained model
    exists.
    """
    ndim = image.ndim

    # ------------------------------------------------------------------
    # 2-D images
    # ------------------------------------------------------------------
    if ndim == 2:
        img_uint8 = _to_uint8(image)
        h = kwargs.get("h", 10)
        template_window_size = kwargs.get("template_window_size", 7)
        search_window_size = kwargs.get("search_window_size", 21)

        try:
            import cv2
            denoised = cv2.fastNlMeansDenoising(
                img_uint8,
                None,
                h=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
            )
            return _restore_dtype(denoised, image.dtype)
        except ImportError:
            logger.warning("cv2 not available, trying scikit-image fallback")

        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma = kwargs.get("sigma")
            if sigma is None:
                sigma = estimate_sigma(img_uint8, channel_axis=None)
            denoised = denoise_nl_means(
                img_uint8,
                h=0.8 * sigma,
                sigma=sigma,
                fast_mode=True,
                patch_size=kwargs.get("patch_size", 5),
                patch_distance=kwargs.get("patch_distance", 6),
                channel_axis=None,
            )
            return _restore_dtype(denoised, image.dtype)
        except ImportError:
            pass

        raise ImportError(
            "cv2 or scikit-image required for denoising. "
            "Install 'opencv-python-headless' or 'scikit-image'."
        )

    # ------------------------------------------------------------------
    # 3-D images
    # ------------------------------------------------------------------
    elif ndim == 3:
        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma = kwargs.get("sigma")
            if sigma is None:
                sigma = estimate_sigma(image, channel_axis=None)

            denoised = denoise_nl_means(
                image,
                h=0.8 * sigma,
                sigma=sigma,
                fast_mode=True,
                patch_size=kwargs.get("patch_size", 5),
                patch_distance=kwargs.get("patch_distance", 6),
                channel_axis=None,
            )
            return denoised
        except ImportError:
            pass

        try:
            import cv2
            # 3-D → process slice-by-slice
            h_val = kwargs.get("h", 10)
            tw = kwargs.get("template_window_size", 7)
            sw = kwargs.get("search_window_size", 21)
            out = np.empty_like(image)
            for idx in range(image.shape[0]):
                out[idx] = cv2.fastNlMeansDenoising(
                    _to_uint8(image[idx]),
                    None,
                    h=h_val,
                    templateWindowSize=tw,
                    searchWindowSize=sw,
                )
            return _restore_dtype(out, image.dtype)
        except ImportError:
            pass

        raise ImportError(
            "scikit-image or OpenCV required for 3-D denoising."
        )

    else:
        raise ValueError(
            f"Expected 2-D or 3-D array, got {ndim}-D with shape {image.shape}"
        )


def _correct_motion(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Correct motion artefacts via a simple rolling-average filter.

    For single 2-D images a spatial Gaussian blur is applied.  For
    3-D stacks (z, y, x) or (t, y, x) the function averages along the
    leading axis.
    """
    ndim = image.ndim
    window = kwargs.get("window", 3)

    # Single 2-D / multi-channel image → spatial blur.
    if ndim == 2 or (ndim == 3 and image.shape[-1] in (1, 3, 4)):
        sigma = kwargs.get("sigma", 1.0)
        try:
            import cv2
            kernel_size = (window, window)
            return cv2.GaussianBlur(image, kernel_size, sigma)
        except ImportError:
            pass

        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=sigma)
        except ImportError:
            pass

        raise ImportError(
            "OpenCV or SciPy required for motion correction. "
            "Install 'opencv-python-headless' or 'scipy'."
        )

    # 3-D volume → temporal / z-axis averaging.
    elif ndim == 3:
        result = image.astype(np.float64)
        half = window // 2
        for i in range(result.shape[0]):
            start = max(0, i - half)
            end = min(result.shape[0], i + half + 1)
            result[i] = np.mean(image[start:end], axis=0)
        return _restore_dtype(result, image.dtype)

    else:
        raise ValueError(
            f"Expected 2-D or 3-D image, got {ndim}-D with shape {image.shape}"
        )


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalise ``array`` to the uint8 range [0, 255]."""
    if array.dtype == np.uint8:
        return array
    info = np.iinfo(array.dtype) if np.issubdtype(array.dtype, np.integer) else None
    f_min = info.min if info is not None else array.min()
    f_max = info.max if info is not None else array.max()
    scaled = ((array.astype(np.float64) - f_min) / (f_max - f_min or 1.0) * 255.0)
    return scaled.clip(0, 255).astype(np.uint8)


def _restore_dtype(
    denoised: np.ndarray,
    original_dtype: np.dtype,
) -> np.ndarray:
    """Cast the denoised result back to the original data type."""
    if denoised.dtype == original_dtype:
        return denoised
    if np.issubdtype(original_dtype, np.integer):
        return denoised.round().clip(
            np.iinfo(original_dtype).min,
            np.iinfo(original_dtype).max,
        ).astype(original_dtype)
    return denoised.astype(original_dtype)
