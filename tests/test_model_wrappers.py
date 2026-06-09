"""Tests for pybnt/processing/model_wrappers.py — pretrained model wrappers."""

import builtins
import sys
from unittest import mock

import pytest


def _insert_fake_module(name):
    """Insert a fake module into sys.modules and return the fake module."""
    fake = mock.MagicMock()
    sys.modules[name] = fake
    return fake


class TestSynthSegWrapper:
    """Tests for SynthSegWrapper."""

    def test_availability_check_import_succeeds(self):
        """is_available returns True when brainseg_containers is importable."""
        _insert_fake_module("brainseg_containers")
        try:
            import pybnt.processing.model_wrappers as mw
            result = mw.SynthSegWrapper.is_available()
            assert result is True
        finally:
            sys.modules.pop("brainseg_containers", None)

    def test_availability_check_import_fails_cli_found(self):
        """is_available returns True when CLI synthseg exists but import fails."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            import pybnt.processing.model_wrappers as mw
            result = mw.SynthSegWrapper.is_available()
            assert result is True

    def test_availability_check_both_fail(self):
        """is_available returns False when neither import nor CLI works."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            import pybnt.processing.model_wrappers as mw
            result = mw.SynthSegWrapper.is_available()
            assert result is False


class TestHDBetWrapper:
    """Tests for HDBetWrapper."""

    def test_availability_check_import_succeeds(self):
        """is_available returns True when hd_bet is importable."""
        _insert_fake_module("hd_bet")
        try:
            import pybnt.processing.model_wrappers as mw
            result = mw.HDBetWrapper.is_available()
            assert result is True
        finally:
            sys.modules.pop("hd_bet", None)

    def test_availability_check_import_fails(self):
        """is_available returns False when hd_bet is not importable."""
        import pybnt.processing.model_wrappers as mw
        result = mw.HDBetWrapper.is_available()
        assert result is False


class TestEDSRPretrained:
    """Tests for EDSRPretrained."""

    def test_availability_check_torch_available(self):
        """is_available returns True when torch is importable."""
        import pybnt.processing.model_wrappers as mw
        result = mw.EDSRPretrained.is_available()
        assert result is True

    def test_availability_check_torch_missing(self):
        """is_available returns False when torch is not importable."""
        orig_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("torch not available")
            return orig_import(name, *a, **kw)

        builtins.__import__ = fake_import
        try:
            import pybnt.processing.model_wrappers as mw
            result = mw.EDSRPretrained.is_available()
            assert result is False
        finally:
            builtins.__import__ = orig_import

    def test_download_weights_already_exists(self, tmp_path):
        """download_weights returns existing path if file already there."""
        import os
        from pybnt.processing.model_wrappers import EDSRPretrained

        weight_file = os.path.join(str(tmp_path), "edsr_x2.pt")
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(weight_file, "w") as f:
            f.write("dummy weights")

        result = EDSRPretrained.download_weights(str(tmp_path))
        assert result == weight_file

    def test_download_weights_first_url_succeeds(self, tmp_path):
        """download_weights uses first URL and returns path on success."""
        import os
        from pybnt.processing.model_wrappers import EDSRPretrained

        target = str(tmp_path)
        weight_file = os.path.join(target, "edsr_x2.pt")

        with mock.patch("wget.download") as mock_dl:
            mock_dl.return_value = weight_file
            result = EDSRPretrained.download_weights(target)
            mock_dl.assert_called_once()
            assert result == weight_file

    def test_download_weights_fallback_after_failure(self, tmp_path):
        """download_weights tries second URL when first fails."""
        import os
        from pybnt.processing.model_wrappers import EDSRPretrained

        target = str(tmp_path)
        weight_file = os.path.join(target, "edsr_x2.pt")

        with mock.patch("wget.download") as mock_dl:
            mock_dl.side_effect = [Exception("download failed"), weight_file]
            result = EDSRPretrained.download_weights(target)
            assert mock_dl.call_count == 2
            assert result == weight_file

    def test_download_weights_all_urls_fail(self, tmp_path):
        """download_weights raises RuntimeError when all URLs fail."""
        from pybnt.processing.model_wrappers import EDSRPretrained

        with mock.patch("wget.download") as mock_dl:
            mock_dl.side_effect = Exception("download failed")
            with pytest.raises(RuntimeError, match="Could not download"):
                EDSRPretrained.download_weights(str(tmp_path))


class TestDownloadAll:
    """Tests for download_all_pretrained helper."""

    def test_download_all_pretrained_success(self, tmp_path):
        """download_all_pretrained succeeds and returns dict with edsr key."""
        import os
        from pybnt.processing.model_wrappers import download_all_pretrained

        target = str(tmp_path)
        edsr_dir = os.path.join(target, "edsr")
        os.makedirs(edsr_dir, exist_ok=True)

        weight_file = os.path.join(edsr_dir, "edsr_x2.pt")
        with open(weight_file, "w") as f:
            f.write("dummy")

        results = download_all_pretrained(target)
        assert "edsr" in results
        assert results["edsr"] == weight_file

    def test_download_all_pretrained_edsr_fails(self, tmp_path):
        """download_all_pretrained handles EDSR download failure gracefully."""
        from pybnt.processing.model_wrappers import download_all_pretrained

        with mock.patch(
            "pybnt.processing.model_wrappers.EDSRPretrained.download_weights",
            side_effect=RuntimeError("download failed"),
        ):
            results = download_all_pretrained(str(tmp_path))
        assert isinstance(results, dict)
        assert "edsr" not in results


class TestAllWrappersInterface:
    """Verify every wrapper has the required interface."""

    def test_all_wrappers_have_is_available_classmethod(self):
        """Every wrapper class exposes is_available() as a staticmethod."""
        from pybnt.processing import model_wrappers

        wrapper_classes = [
            model_wrappers.SynthSegWrapper,
            model_wrappers.HDBetWrapper,
            model_wrappers.EDSRPretrained,
        ]

        for cls in wrapper_classes:
            method = getattr(cls, "is_available", None)
            assert method is not None, (
                f"{cls.__name__} missing is_available()"
            )
            assert callable(method), (
                f"{cls.__name__}.is_available is not callable"
            )
