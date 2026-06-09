"""Functional connectivity analysis using nilearn."""
import os
import tempfile
import warnings

import nibabel as nib
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiSpheresMasker
from nilearn.maskers import NiftiMasker


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 255] range using min-max normalization."""
    import cv2
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def _compute_correlation(fmri_filename: str, ratio: float = 1.0) -> np.ndarray:
    """Core correlation computation - shared by all entry points.

    Computes a connectivity matrix from a 4D fMRI nifti file using
    the posterior cingulate cortex as a seed region.

    Parameters
    ----------
    fmri_filename : str
        Path to a 4D fMRI NIfTI file.
    ratio : float
        Smoothing/voxel ratio derived from image dimensions.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    seed_coords = [(0, -52, 18)]
    seed_masker = NiftiSpheresMasker(
        seed_coords, radius=8 * ratio,
        detrend=True, standardize=True,
    )

    brain_masker = NiftiMasker(
        smoothing_fwhm=6 * ratio,
        detrend=True, standardize=True,
    )
    brain_time_series = brain_masker.fit_transform(fmri_filename)

    connectome_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = connectome_measure.fit_transform([brain_time_series])[0]

    print("Cor_matrix_shape", correlation_matrix.shape)
    print(correlation_matrix)
    return correlation_matrix


def compute_correlation_matrix(
    image_input,
    input_type: str = "auto",
) -> np.ndarray:
    """Compute correlation matrix from fMRI data.

    Parameters
    ----------
    image_input : str or np.ndarray or nib.Nifti1Image
        Input data. Supported types:
        - str ending with .nii/.nii.gz: 4D NIfTI file path
        - str (directory): directory of image files
        - nib.Nifti1Image / nib.Nifti2Image: nifti object
        - np.ndarray: raw 4D array
    input_type : str
        One of "auto", "path_4d", "path_dir", "nifti_obj", "array".
        "auto" detects type automatically.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    try:
        # Directory input
        if input_type == "path_dir" or (
            input_type == "auto"
            and isinstance(image_input, str)
            and not (
                image_input.endswith('.nii')
                or image_input.endswith('.nii.gz')
            )
        ):
            return compute_connectivity_from_directory(image_input)

        # Nifti object or raw array input
        if input_type in ("nifti_obj", "array") or (
            input_type == "auto"
            and not isinstance(image_input, str)
        ):
            if isinstance(image_input, (nib.Nifti1Image, nib.Nifti2Image)):
                data = image_input.get_fdata()
            else:
                data = np.asarray(image_input)

            ratio = data.shape[0] // 50
            img4d_nib = nib.Nifti1Image(data, np.eye(4))

            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tf:
                nib.save(img4d_nib, tf.name)
                try:
                    return _compute_correlation(tf.name, ratio=ratio)
                finally:
                    os.unlink(tf.name)

        # String path — treat as 4D nifti
        if input_type == "path_4d" or input_type == "auto":
            try:
                video = nib.load(image_input).get_fdata()
            except Exception as e:
                raise ValueError(f"Cannot load NIfTI from {image_input}: {e}") from e

            if len(video.shape) != 4:
                raise ValueError(
                    "Shape error: input is not a 4D fMRI image "
                    f"(got {len(video.shape)}D)"
                )
            ratio = video.shape[0] // 50
            return _compute_correlation(image_input, ratio=ratio)

        raise ValueError(f"Unknown input_type: {input_type}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to compute correlation matrix: {e}"
        ) from e


def compute_connectivity_from_directory(image_dir: str) -> np.ndarray:
    """Compute connectivity matrix from a directory of images.

    Reads all images from a directory, normalizes them, stacks into
    a 4D volume, and computes the correlation matrix.

    Parameters
    ----------
    image_dir : str
        Path to directory containing image files (.nii, .nii.gz, or
        standard image formats readable by cv2).

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    try:
        img_files = sorted(os.listdir(image_dir))
    except OSError as e:
        raise RuntimeError(f"Cannot read directory {image_dir}: {e}") from e

    if not img_files:
        raise ValueError(f"No files found in directory: {image_dir}")

    import cv2

    images = []
    for filename in img_files:
        filepath = os.path.join(image_dir, filename)
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            img = nib.load(filepath).get_fdata()
        else:
            img = cv2.imread(filepath)
            if img is None:
                warnings.warn(f"Skipping non-image file: {filename}")
                continue
        img = _normalize(img)
        images.append(img)

    if not images:
        raise ValueError(f"No valid images found in {image_dir}")

    img4d = np.array(images)
    ratio = img4d.shape[0] // 50
    img4d_nib = nib.Nifti1Image(img4d, np.eye(4))

    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tf:
        nib.save(img4d_nib, tf.name)
        try:
            return _compute_correlation(tf.name, ratio=ratio)
        finally:
            os.unlink(tf.name)
