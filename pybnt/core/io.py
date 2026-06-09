"""
I/O utilities for DICOM ↔ NIfTI ↔ image conversions.

Provides functions to convert between DICOM files, NIfTI volumes,
and standard image formats (PNG, JPEG, etc.).
"""

from typing import Optional

import cv2 as cv
import nibabel as nib
import numpy as np
import pydicom as dic  # type: ignore[import-untyped]

from pybnt.core.imageproc import normalize


def dicom_to_nifti(dpath: str, npath: str) -> None:
    """Convert a DICOM file to a NIfTI volume.

    The pixel data is normalised to [0, 255] before saving.

    Args:
        dpath: Path to the input DICOM (``.dcm``) file.
        npath: Path for the output NIfTI (``.nii`` / ``.nii.gz``) file.
    """
    ds = dic.dcmread(dpath)
    img = ds.pixel_array
    img = normalize(img)
    nib.save(nib.Nifti1Image(img, np.eye(4)), npath)


def dicom_to_image(dpath: str, ipath: str) -> None:
    """Convert a DICOM file to a standard image file.

    Args:
        dpath: Path to the input DICOM file.
        ipath: Path for the output image (e.g. ``.png``).
    """
    ds = dic.dcmread(dpath)
    img = ds.pixel_array
    img = normalize(img)
    cv.imwrite(ipath, img)


def nifti_to_image(
    npath: str,
    ipath: Optional[str] = None,
    normalize_flag: bool = False,
) -> Optional[np.ndarray]:
    """Convert a NIfTI volume to an image array or file.

    Args:
        npath: Path to the input NIfTI file.
        ipath: If provided, the image is written to this path.
        normalize_flag: If ``True``, normalise the data to [0, 255].

    Returns:
        Image array if *ipath* is ``None``, otherwise ``None``.
    """
    img = nib.load(npath).get_fdata()
    img = np.array(img)
    if normalize_flag:
        img = normalize(img)
    if ipath is not None:
        cv.imwrite(ipath, img)
        return None
    return img


def image_to_nifti(img: np.ndarray, npath: str) -> None:
    """Convert an image array to a NIfTI volume.

    The image is normalised to [0, 255] before saving.

    Args:
        img: Image array.
        npath: Path for the output NIfTI file.
    """
    img = normalize(img)
    nib.save(nib.Nifti1Image(img, np.eye(4)), npath)


def nifti_shape(npath: str) -> tuple[int, ...]:
    """Return the shape of a NIfTI volume.

    Args:
        npath: Path to the NIfTI file.

    Returns:
        Shape tuple (e.g. ``(256, 256, 100)``).
    """
    return np.array(nib.load(npath).get_fdata()).shape


# Backward-compatible aliases
d2nif = dicom_to_nifti
d2img = dicom_to_image
n2img = nifti_to_image
img2n = image_to_nifti
nshape = nifti_shape
