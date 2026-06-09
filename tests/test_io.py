"""Tests for pybnt/core/io.py — DICOM/NIfTI I/O utilities."""

import os
import pytest
import numpy as np
from pybnt.core.io import (
    image_to_nifti,
    nifti_to_image,
    nifti_shape,
    dicom_to_nifti,
    dicom_to_image,
)


class TestNiftiIO:
    """Tests for NIfTI read/write utilities."""

    def test_nifti_roundtrip(self, tmp_path):
        """Test writing and reading a NIfTI file preserves shape."""
        data = np.random.rand(10, 10, 10).astype(np.float32)
        path = str(tmp_path / "test_roundtrip.nii")
        image_to_nifti(data, path)
        assert os.path.exists(path), "NIfTI file should be created"
        shape = nifti_shape(path)
        assert shape == (10, 10, 10), (
            f"Expected shape (10, 10, 10), got {shape}"
        )

    def test_nifti_to_image_returns_array(self, tmp_path):
        """Test reading NIfTI file returns numpy array."""
        data = np.ones((5, 5, 5), dtype=np.float32)
        path = str(tmp_path / "test_read.nii")
        image_to_nifti(data, path)
        read_data = nifti_to_image(path)
        assert isinstance(read_data, np.ndarray), "Should return numpy array"
        assert read_data.shape == (5, 5, 5), (
            f"Expected (5, 5, 5), got {read_data.shape}"
        )

    def test_nifti_to_image_with_normalize(self, tmp_path):
        """Test reading NIfTI with normalization flag returns non-None array."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
        path = str(tmp_path / "test_norm.nii")
        image_to_nifti(data, path)
        read_data = nifti_to_image(path, normalize_flag=True)
        assert read_data is not None
        assert np.all(read_data > -1e-3), "Normalized data should be approximately >= 0"
        assert isinstance(read_data, np.ndarray)

    def test_nifti_to_image_write_to_file(self, tmp_path):
        """Test reading NIfTI and writing to an image file."""
        data = np.ones((3, 3, 3), dtype=np.float32) * 128
        nii_path = str(tmp_path / "test_write.nii")
        img_path = str(tmp_path / "output.png")
        image_to_nifti(data, nii_path)
        result = nifti_to_image(nii_path, ipath=img_path, normalize_flag=True)
        assert result is None, "Should return None when ipath is provided"
        assert os.path.exists(img_path), "Image output file should be created"

    def test_nifti_shape_returns_tuple(self, tmp_path):
        """Test nifti_shape returns a tuple."""
        data = np.ones((3, 4, 5), dtype=np.float32)
        path = str(tmp_path / "test_shape.nii")
        image_to_nifti(data, path)
        shape = nifti_shape(path)
        assert isinstance(shape, tuple), "Shape should be a tuple"
        assert shape == (3, 4, 5)

    def test_image_to_nifti_2d_input(self, tmp_path):
        """Test converting a 2D image to NIfTI."""
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
        path = str(tmp_path / "test_2d.nii")
        image_to_nifti(data, path)
        assert os.path.exists(path)
        shape = nifti_shape(path)
        assert len(shape) == 2, f"2D image should produce 2D NIfTI, got {len(shape)}D"

    def test_image_to_nifti_normalizes_output(self, tmp_path):
        """Test image_to_nifti normalizes data close to [0, 255] range."""
        data = np.array([[-100.0, 0.0], [100.0, 200.0]], dtype=np.float32)
        path = str(tmp_path / "test_norm_out.nii")
        image_to_nifti(data, path)
        read_data = nifti_to_image(path)
        assert read_data is not None
        assert read_data.min() > -1e-3, "Data should be normalized to near [0, 255]"
        assert read_data.max() <= 255.001, f"Max should be close to 255, got {read_data.max()}"


class TestDicomIO:
    """Tests for DICOM conversion utilities (using existing test DICOM)."""

    def test_dicom_file_exists(self):
        """Test that a test DICOM file is available in the project root."""
        dcm_path = "00000005.dcm"
        if not os.path.exists(dcm_path):
            pytest.skip("Test DICOM file 00000005.dcm not found")

    def test_dicom_to_nifti(self, tmp_path):
        """Test converting DICOM to NIfTI."""
        dcm_path = "00000005.dcm"
        if not os.path.exists(dcm_path):
            pytest.skip("Test DICOM file not found")
        nii_path = str(tmp_path / "test_dicom.nii")
        dicom_to_nifti(dcm_path, nii_path)
        assert os.path.exists(nii_path), "NIfTI file should be created"

    def test_dicom_to_image(self, tmp_path):
        """Test converting DICOM to PNG image."""
        dcm_path = "00000005.dcm"
        if not os.path.exists(dcm_path):
            pytest.skip("Test DICOM file not found")
        img_path = str(tmp_path / "test_dicom.png")
        dicom_to_image(dcm_path, img_path)
        assert os.path.exists(img_path), "Image file should be created"

    def test_dicom_to_nifti_shape(self, tmp_path):
        """Test DICOM-to-NIfTI conversion preserves some data."""
        dcm_path = "00000005.dcm"
        if not os.path.exists(dcm_path):
            pytest.skip("Test DICOM file not found")
        nii_path = str(tmp_path / "test_dicom_shape.nii")
        dicom_to_nifti(dcm_path, nii_path)
        shape = nifti_shape(nii_path)
        assert len(shape) >= 2, "Should produce at least a 2D image"
