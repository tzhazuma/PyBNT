"""Vision-Language Model integration for brain MRI analysis.

Provides:
- BrainMRI SigLIP: 3D brain MRI embedding extraction (1152-dim)
- MiMo-V2.5 multimodal: image description via OpenCodeGo

Optional dependencies: transformers, torch (only needed for SigLIP)
"""

import numpy as np
from pathlib import Path

from pybnt.core.logconf import logger


class BrainMRISigLIP:
    """BrainMRI SigLIP visual encoder for 3D brain MRI scans.

    Extracts 1152-dimensional embeddings from NIfTI files.
    Model: shenxiaochen/brain-mri-siglip on HuggingFace (0.9B params).

    Requires: pip install transformers torch
    """

    def __init__(self, model_name: str = "shenxiaochen/brain-mri-siglip",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    @property
    def is_available(self) -> bool:
        """Check if transformers and torch are available."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        """Load the model (lazy, on first use)."""
        if self._model is not None:
            return
        if not self.is_available:
            raise ImportError(
                "transformers and torch are required for BrainMRI SigLIP. "
                "Install with: pip install transformers torch"
            )
        from transformers import AutoModel, AutoImageProcessor

        logger.info("Loading BrainMRI SigLIP model: %s", self.model_name)
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    def extract_embeddings(self, nifti_path: str) -> np.ndarray:
        """Extract 1152-dimensional embedding from a NIfTI file.

        Args:
            nifti_path: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            1152-dim numpy array
        """
        self._load()
        import nibabel as nib

        img = nib.load(nifti_path)
        data = img.get_fdata()

        # Normalize and prepare for model
        import torch
        inputs = self._processor(data, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Get pooled embedding
        embedding = outputs.pooler_output.cpu().numpy().flatten()
        return embedding

    def close(self):
        """Release model memory."""
        self._model = None
        self._processor = None
        import gc
        gc.collect()


def describe_brain_image(
    image_path: str,
    prompt: str = "Describe this brain MRI scan in detail. Identify any visible structures, "
                   "the imaging modality, and any notable findings.",
    provider: str = "opencodego",
    model: str = "mimo-v2.5",
) -> str | None:
    """Send a brain image for AI-powered multimodal description.

    Uses MiMo-V2.5 (multimodal) via OpenCodeGo provider.
    Falls back to SigLIP embedding + text description if MiMo unavailable.

    Args:
        image_path: Path to image file (NIfTI, PNG, JPG, etc.)
        prompt: Description prompt
        provider: LLM provider (must be "opencodego" for multimodal)
        model: Model name

    Returns:
        Description text or None
    """
    from pybnt.ai.llm import ask_llm_multimodal
    return ask_llm_multimodal(image_path, prompt, provider, model)


def extract_embeddings_to_file(
    nifti_path: str,
    output_path: str | None = None,
) -> str:
    """Extract embeddings and save to .npy file.

    Args:
        nifti_path: Path to NIfTI file
        output_path: Output path (default: <stem>_embedding.npy)

    Returns:
        Path to saved embedding file
    """
    if output_path is None:
        output_path = Path(nifti_path).stem + "_embedding.npy"

    model = BrainMRISigLIP()
    embedding = model.extract_embeddings(nifti_path)
    np.save(output_path, embedding)
    logger.info("Embedding saved to %s (shape: %s)", output_path, embedding.shape)
    return output_path
