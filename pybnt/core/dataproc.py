"""
Data processing utilities for train/test splitting.

Provides a function to split image datasets from a directory
(or array) into training and testing subsets, with optional
NIfTI-to-2D conversion and normalization.
"""

from typing import Union

import cv2 as cv
import os

from pybnt.core.imageproc import normalize, to_2d
from pybnt.core.io import nifti_to_image


def split_train_test(
    imgpath: Union[str, list],
    trainpath: str,
    testpath: str,
    split_type: str = "path",
    ratio: float = 0.8,
    nifti: bool = False,
) -> None:
    """Split an image dataset into training and testing sets.

    Images are normalised before being written to the output directories.

    Args:
        imgpath: Path to a directory of images (when ``split_type="path"``)
                 or a list of image arrays (when ``split_type="array"``).
        trainpath: Directory to write training images.
        testpath: Directory to write testing images.
        split_type: ``"path"`` reads from disk; ``"array"`` uses the
                    provided list.
        ratio: Fraction of data to use for training (default 0.8).
        nifti: If ``True``, treat inputs as NIfTI volumes and convert
               each 3D volume into 2D slices before splitting.
    """
    if split_type == "path":
        assert isinstance(imgpath, str)
        all_paths = os.listdir(imgpath)
        split_idx = int(len(all_paths) * ratio)
        train_paths = all_paths[:split_idx]
        test_paths = all_paths[split_idx:]

        for filename in train_paths:
            if nifti:
                vol = nifti_to_image(os.path.join(imgpath, filename))
                slices = to_2d(vol, 2)
                for img in slices:
                    img = normalize(img)
                    cv.imwrite(os.path.join(trainpath, filename), img)
            else:
                img = cv.imread(os.path.join(imgpath, filename), cv.IMREAD_GRAYSCALE)
                img = normalize(img)
                cv.imwrite(os.path.join(trainpath, filename), img)

        for filename in test_paths:
            if nifti:
                vol = nifti_to_image(os.path.join(imgpath, filename))
                slices = to_2d(vol, 2)
                for img in slices:
                    img = normalize(img)
                    cv.imwrite(os.path.join(testpath, filename), img)
            else:
                img = cv.imread(os.path.join(imgpath, filename), cv.IMREAD_GRAYSCALE)
                img = normalize(img)
                cv.imwrite(os.path.join(testpath, filename), img)

    elif split_type == "array":
        assert isinstance(imgpath, list)
        split_idx = int(len(imgpath) * ratio)
        train_data = imgpath[:split_idx]
        test_data = imgpath[split_idx:]

        for arr in train_data:
            arr = normalize(arr)
            cv.imwrite(os.path.join(trainpath, str(id(arr))), arr)

        for arr in test_data:
            arr = normalize(arr)
            cv.imwrite(os.path.join(testpath, str(id(arr))), arr)
