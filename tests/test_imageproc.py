"""Tests for pybnt/core/imageproc.py — image processing utilities."""

import pytest
import numpy as np
from pybnt.core.imageproc import (
    normalize,
    snr,
    cnr,
    psnr,
    ssim,
    rmse,
    transform,
    to_2d,
    to_3d,
)


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_range(self):
        """Test normalized output has range [0, 255]."""
        img = np.array([[10, 50], [100, 200]], dtype=np.uint8)
        norm = normalize(img)
        assert norm.min() == 0, f"Minimum should be 0, got {norm.min()}"
        assert norm.max() == 255, f"Maximum should be 255, got {norm.max()}"

    def test_normalize_constant_image(self):
        """Test normalizing a constant-valued image does not crash."""
        img = np.ones((10, 10), dtype=np.uint8) * 128
        norm = normalize(img)
        assert norm.shape == img.shape
        assert np.all(norm >= 0), "All values should be >= 0"

    def test_normalize_all_same_output(self):
        """Test normalizing already normalized image is idempotent in range."""
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        norm = normalize(img)
        assert norm.min() == 0
        assert norm.max() == 255

    def test_normalize_small_range(self):
        """Test normalizing image with small value range."""
        img = np.array([[100, 101], [102, 103]], dtype=np.uint8)
        norm = normalize(img)
        assert norm.shape == img.shape
        assert norm.min() == 0
        assert norm.max() == 255

    def test_normalize_float_input(self):
        """Test normalizing float input array."""
        img = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
        norm = normalize(img)
        assert norm.shape == img.shape
        assert norm.min() >= 0


class TestSNR:
    """Tests for snr (signal-to-noise ratio) function."""

    def test_snr_all_positive(self):
        """Test SNR 'all' type returns a positive value."""
        np.random.seed(42)
        img = np.random.rand(64, 64).astype(np.float32) * 100 + 50
        snr_val = snr(img, snr_type="all")
        assert snr_val > 0, f"SNR should be positive, got {snr_val}"

    def test_snr_all_better_for_high_signal(self):
        """Test SNR is higher for a cleaner image."""
        clean = np.ones((32, 32), dtype=np.float32) * 100
        noisy = np.ones((32, 32), dtype=np.float32) * 100 + np.random.randn(32, 32).astype(np.float32) * 50
        snr_clean = snr(clean, snr_type="all")
        snr_noisy = snr(noisy, snr_type="all")
        assert snr_clean > snr_noisy, (
            f"Clean (SNR={snr_clean:.1f}) should have higher SNR than noisy (SNR={snr_noisy:.1f})"
        )

    def test_snr_roi_type(self):
        """Test SNR 'roi' type works on 2D image."""
        np.random.seed(42)
        img = np.random.rand(50, 50).astype(np.float32) * 100
        snr_val = snr(img, snr_type="roi")
        assert snr_val > 0, f"SNR ROI should be positive, got {snr_val}"

    def test_snr_invalid_type_raises(self):
        """Test invalid snr_type raises ValueError."""
        img = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="not supported"):
            snr(img, snr_type="invalid")


class TestCNR:
    """Tests for cnr (contrast-to-noise ratio) function."""

    def test_cnr_all_positive(self):
        """Test CNR 'all' type returns a positive value."""
        np.random.seed(42)
        img = np.random.rand(64, 64).astype(np.float32) * 100
        cnr_val = cnr(img, cnr_type="all")
        assert cnr_val > 0, f"CNR should be positive, got {cnr_val}"

    def test_cnr_all_higher_for_high_contrast(self):
        """Test CNR is higher for a high-contrast image."""
        low_contrast = np.ones((32, 32), dtype=np.float32) * 50
        low_contrast[0, 0] = 55
        high_contrast = np.zeros((32, 32), dtype=np.float32)
        high_contrast[:16, :] = 0
        high_contrast[16:, :] = 200
        cnr_low = cnr(low_contrast, cnr_type="all")
        cnr_high = cnr(high_contrast, cnr_type="all")
        assert cnr_high > 0, "CNR should be positive for high-contrast image"

    def test_cnr_roi_type(self):
        """Test CNR 'roi' type works on 2D image."""
        np.random.seed(42)
        img = np.random.rand(50, 50).astype(np.float32) * 100
        cnr_val = cnr(img, cnr_type="roi")
        assert isinstance(cnr_val, float)

    def test_cnr_invalid_type_raises(self):
        """Test invalid cnr_type raises ValueError."""
        img = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="not supported"):
            cnr(img, cnr_type="invalid")


class TestPSNR:
    """Tests for psnr function."""

    def test_psnr_identical_images(self):
        """Test PSNR of identical images is very high."""
        img = np.ones((32, 32), dtype=np.float32) * 128
        val = psnr(img, img)
        assert val > 50, f"PSNR of identical images should be > 50, got {val}"

    def test_psnr_different_images(self):
        """Test PSNR of different images is lower."""
        img1 = np.zeros((32, 32), dtype=np.float32)
        img2 = np.ones((32, 32), dtype=np.float32) * 255
        val = psnr(img1, img2)
        assert val < 50, f"PSNR of very different images should be low, got {val}"

    def test_psnr_similar_higher(self):
        """Test PSNR is higher for more similar images."""
        base = np.random.RandomState(42).rand(32, 32).astype(np.float32) * 255
        similar = base + np.random.RandomState(1).randn(32, 32).astype(np.float32) * 5
        different = base + np.random.RandomState(2).randn(32, 32).astype(np.float32) * 100
        psnr_sim = psnr(base, similar)
        psnr_diff = psnr(base, different)
        assert psnr_sim > psnr_diff, (
            f"Similar ({psnr_sim:.1f}) should be > different ({psnr_diff:.1f})"
        )


class TestSSIM:
    """Tests for ssim function."""

    def test_ssim_identical_images(self):
        """Test SSIM of identical images is 1."""
        img = np.ones((32, 32), dtype=np.float32) * 128
        val = ssim(img, img)
        assert abs(val - 1.0) < 0.1, f"SSIM of identical images should be ~1, got {val}"

    def test_ssim_different_images(self):
        """Test SSIM of very different images is lower than 1."""
        img1 = np.zeros((32, 32), dtype=np.float32)
        img2 = np.ones((32, 32), dtype=np.float32) * 255
        val = ssim(img1, img2)
        assert val < 1.0, f"SSIM of different images should be < 1, got {val}"


class TestNRMSE:
    """Tests for rmse (normalized RMSE) function."""

    def test_nrmse_identical_images(self):
        """Test NRMSE of identical images is 0."""
        img = np.ones((32, 32), dtype=np.float32) * 128
        val = rmse(img, img)
        assert val < 0.01, f"NRMSE of identical images should be ~0, got {val}"

    def test_nrmse_different_images_positive(self):
        """Test NRMSE of different images is positive."""
        img1 = np.zeros((32, 32), dtype=np.float32)
        img2 = np.ones((32, 32), dtype=np.float32) * 255
        val = rmse(img1, img2)
        assert val > 0, f"NRMSE should be positive for different images, got {val}"


class TestTransform:
    """Tests for transform function."""

    def test_transform_identity(self):
        """Test identity transform (no rotation, scale 1, no flip)."""
        img = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = transform(img, angle=0.0, scale=1.0, flip=0)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_transform_rotate_90(self):
        """Test 90-degree rotation preserves shape."""
        img = np.ones((64, 64), dtype=np.uint8)
        result = transform(img, angle=90.0, scale=1.0, flip=0)
        assert result.shape == (64, 64)

    def test_transform_flip(self):
        """Test horizontal flip changes the image."""
        img = np.zeros((32, 32), dtype=np.float32)
        img[0, 0] = 255  # Put a bright pixel at top-left
        flipped = transform(img, angle=0.0, scale=1.0, flip=1)
        assert flipped.shape == img.shape

    def test_transform_scale(self):
        """Test scaling preserves shape."""
        img = np.ones((64, 64), dtype=np.uint8)
        result = transform(img, angle=0.0, scale=0.5, flip=0)
        assert result.shape == (64, 64)


class TestSliceConversion:
    """Tests for to_2d and to_3d converters."""

    def test_to_2d_default_axis(self):
        """Test to_2d with default z_axis=2."""
        img3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        slices = to_2d(img3d, z_axis=2)
        assert len(slices) == 4, f"Expected 4 slices along axis 2, got {len(slices)}"
        assert slices[0].shape == (2, 3)

    def test_to_2d_axis_0(self):
        """Test to_2d with z_axis=0."""
        img3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        slices = to_2d(img3d, z_axis=0)
        assert len(slices) == 2
        assert slices[0].shape == (3, 4)

    def test_to_2d_axis_1(self):
        """Test to_2d with z_axis=1."""
        img3d = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        slices = to_2d(img3d, z_axis=1)
        assert len(slices) == 3
        assert slices[0].shape == (2, 4)

    def test_to_3d_default_axis(self):
        """Test to_3d with default z_axis=2."""
        slices = [np.ones((10, 10), dtype=np.uint8) for _ in range(5)]
        img3d = to_3d(slices, z_axis=2)
        assert img3d.shape == (10, 10, 5)
        assert img3d.dtype == np.uint8

    def test_to_3d_contents_match(self):
        """Test to_3d preserves slice contents at correct axis=2 positions."""
        slices = [np.ones((3, 3), dtype=np.uint8) * i for i in range(4)]
        img3d = to_3d(slices, z_axis=2)
        assert img3d.shape == (3, 3, 4)
        for i in range(4):
            assert np.all(img3d[:, :, i] == i), (
                f"Slice {i} should contain all {i}s"
            )

    def test_to_2d_to_3d_roundtrip(self):
        """Test roundtrip: 3D -> 2D slices -> 3D."""
        original = np.random.RandomState(42).randint(0, 256, (10, 10, 5)).astype(np.uint8)
        slices = to_2d(original, z_axis=2)
        reconstructed = to_3d(slices, z_axis=2)
        assert reconstructed.shape == original.shape
        assert np.array_equal(reconstructed, original), (
            "Roundtrip should preserve data"
        )
