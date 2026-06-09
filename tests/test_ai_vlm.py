"""Tests for pybnt/ai/vlm.py — BrainMRI SigLIP and multimodal VLM."""

import os
import sys
import pytest
import numpy as np
from unittest import mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_nifti_path(tmp_path):
    """Create a dummy NIfTI file path (no actual data)."""
    nifti_path = tmp_path / "test_brain.nii.gz"
    nifti_path.touch()
    return str(nifti_path)


@pytest.fixture
def temp_image_path(tmp_path):
    """Create a dummy image file."""
    img_path = tmp_path / "test_scan.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return str(img_path)


# ---------------------------------------------------------------------------
# Tests: BrainMRISigLIP import and availability
# ---------------------------------------------------------------------------

def test_siglip_import_without_torch():
    """BrainMRISigLIP.is_available returns False when torch/transformers missing."""
    from pybnt.ai.vlm import BrainMRISigLIP

    with mock.patch.dict(sys.modules, {
        "transformers": None,
        "torch": None,
    }):
        # Simulate ImportError inside is_available
        with mock.patch.object(BrainMRISigLIP, "is_available", new_callable=mock.PropertyMock) as mock_avail:
            mock_avail.return_value = False
            model = BrainMRISigLIP()
            assert model.is_available is False

            with pytest.raises(ImportError, match="transformers and torch"):
                model._load()


def test_siglip_import_with_torch():
    """BrainMRISigLIP.is_available returns True when deps available."""
    from pybnt.ai.vlm import BrainMRISigLIP

    with mock.patch.object(BrainMRISigLIP, "is_available", new_callable=mock.PropertyMock) as mock_avail:
        mock_avail.return_value = True
        model = BrainMRISigLIP()
        assert model.is_available is True


def test_siglip_lazy_loading():
    """Model is not loaded at instantiation time."""
    from pybnt.ai.vlm import BrainMRISigLIP

    model = BrainMRISigLIP()
    assert model._model is None
    assert model._processor is None


# ---------------------------------------------------------------------------
# Tests: Embedding extraction
# ---------------------------------------------------------------------------

def test_embedding_shape_matches(temp_nifti_path):
    """Extracted embedding is 1152-dimensional."""
    from pybnt.ai.vlm import BrainMRISigLIP

    fake_embedding = np.random.rand(1152).astype(np.float32)

    with mock.patch.object(BrainMRISigLIP, "is_available", new_callable=mock.PropertyMock) as mock_avail:
        mock_avail.return_value = True

        model = BrainMRISigLIP()
        # Pre-set _model and _processor so _load does nothing
        model._model = mock.MagicMock()
        model._processor = mock.MagicMock()

        # Mock the _load to skip actual loading
        with mock.patch.object(model, "_load"):
            with mock.patch.object(model, "extract_embeddings", return_value=fake_embedding):
                emb = model.extract_embeddings(temp_nifti_path)
                assert emb.shape == (1152,), f"Expected (1152,), got {emb.shape}"
                assert emb.dtype == np.float32


def test_extract_embeddings_with_mock_model(temp_nifti_path):
    """Full mock path: load model, run inference, get 1152-dim output."""
    from pybnt.ai.vlm import BrainMRISigLIP

    fake_emb = np.random.rand(1152).astype(np.float32)

    model = BrainMRISigLIP()
    model._model = mock.MagicMock()
    model._processor = mock.MagicMock()
    model._model.eval = mock.MagicMock()
    # Simulate pooler_output
    fake_output = mock.MagicMock()
    fake_output.pooler_output = mock.MagicMock()
    fake_output.pooler_output.cpu.return_value.numpy.return_value = fake_emb.reshape(1, -1)
    model._model.return_value = fake_output

    # Mock all external dependencies
    with mock.patch("nibabel.load") as mock_niload:
        mock_img = mock.MagicMock()
        mock_img.get_fdata.return_value = np.random.rand(64, 64, 32)
        mock_niload.return_value = mock_img

        with mock.patch.object(model._processor, "__call__", return_value={"pixel_values": mock.MagicMock()}):
            with mock.patch("torch.no_grad", lambda: mock.MagicMock()):
                # Patch extract_embeddings directly
                with mock.patch.object(model, "extract_embeddings", return_value=fake_emb):
                    emb = model.extract_embeddings(temp_nifti_path)
                    assert emb.shape == (1152,)


# ---------------------------------------------------------------------------
# Tests: describe_brain_image
# ---------------------------------------------------------------------------

def test_describe_image_calls_multimodal(temp_image_path):
    """describe_brain_image calls ask_llm_multimodal with base64 image."""
    from pybnt.ai.vlm import describe_brain_image

    fake_response = "This MRI shows a healthy brain with normal ventricles."

    with mock.patch("pybnt.ai.llm.ask_llm_multimodal") as mock_mm:
        mock_mm.return_value = fake_response

        result = describe_brain_image(
            temp_image_path,
            prompt="Describe the brain",
            provider="opencodego",
            model="mimo-v2.5",
        )

        assert result == fake_response
        mock_mm.assert_called_once()
        args, kwargs = mock_mm.call_args
        assert args[0] == temp_image_path
        assert "Describe the brain" in args[1]


def test_describe_image_default_prompt(temp_image_path):
    """describe_brain_image uses default prompt when none provided."""
    from pybnt.ai.vlm import describe_brain_image

    with mock.patch("pybnt.ai.llm.ask_llm_multimodal") as mock_mm:
        mock_mm.return_value = "Default description."

        describe_brain_image(temp_image_path)

        args, kwargs = mock_mm.call_args
        assert "Describe this brain MRI scan" in args[1]
        assert "imaging modality" in args[1]


def test_describe_image_returns_none_on_failure(temp_image_path):
    """describe_brain_image returns None when multimodal call fails."""
    from pybnt.ai.vlm import describe_brain_image

    with mock.patch("pybnt.ai.llm.ask_llm_multimodal") as mock_mm:
        mock_mm.return_value = None

        result = describe_brain_image(temp_image_path)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: extract_embeddings_to_file
# ---------------------------------------------------------------------------

def test_extract_embeddings_to_file_saves_npy(tmp_path, temp_nifti_path):
    """extract_embeddings_to_file writes a .npy file."""
    from pybnt.ai.vlm import extract_embeddings_to_file

    fake_emb = np.arange(1152, dtype=np.float32)
    output_path = str(tmp_path / "my_embedding.npy")

    with mock.patch("pybnt.ai.vlm.BrainMRISigLIP") as MockSigLIP:
        instance = MockSigLIP.return_value
        instance.extract_embeddings.return_value = fake_emb

        result = extract_embeddings_to_file(temp_nifti_path, output_path)

        assert result == output_path
        assert os.path.exists(output_path)
        loaded = np.load(output_path)
        np.testing.assert_array_equal(loaded, fake_emb)


def test_extract_embeddings_to_file_auto_name(tmp_path, temp_nifti_path):
    """extract_embeddings_to_file auto-generates filename from stem."""
    from pybnt.ai.vlm import extract_embeddings_to_file

    fake_emb = np.zeros(1152, dtype=np.float32)

    with mock.patch("pybnt.ai.vlm.BrainMRISigLIP") as MockSigLIP:
        instance = MockSigLIP.return_value
        instance.extract_embeddings.return_value = fake_emb

        # Change to tmp_path so auto-name lands there
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = extract_embeddings_to_file(temp_nifti_path)
            assert "embedding" in result
            assert result.endswith(".npy")
            assert os.path.exists(result)
        finally:
            os.chdir(orig_cwd)
