"""Unified wrappers for pretrained brain imaging models.

Provides lazy-loaded interfaces for models that require NO training data:
- SynthSeg: brain segmentation via brainseg-containers
- HD-BET: skull stripping via brainles-hd-bet
- EDSR: super-resolution with pretrained weights download

Each wrapper handles optional dependency imports and auto-downloading.
"""

import os
import subprocess
from pathlib import Path

from pybnt.core.logconf import logger


class SynthSegWrapper:
    """SynthSeg brain segmentation (any contrast, any resolution).

    Wraps brainseg-containers (SynthSeg via Apptainer) or direct SynthSeg.
    No training data needed. Outputs FreeSurfer-format segmentation.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if SynthSeg or brainseg-containers is available."""
        try:
            import brainseg_containers  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            result = subprocess.run(["which", "synthseg"], capture_output=True,
                                    text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def segment(input_path: str, output_path: str | None = None,
                device: str = "cpu") -> str:
        """Segment brain into anatomical regions.

        Args:
            input_path: Path to NIfTI file
            output_path: Path for output segmentation (default: <input>_seg.nii.gz)
            device: "cpu" or "cuda"

        Returns:
            Path to segmentation output
        """
        if output_path is None:
            output_path = str(Path(input_path).stem) + "_synthseg.nii.gz"

        try:
            from brainseg_containers import segment
            segment(input_path, output_path, backend="synthseg", device=device)
            logger.info("SynthSeg segmentation complete: %s", output_path)
            return str(output_path)
        except ImportError:
            pass

        try:
            cmd = ["synthseg", "--i", input_path, "--o", output_path,
                   "--device", device]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("SynthSeg segmentation complete: %s", output_path)
            return str(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "SynthSeg not available. Install with: "
                "pip install brainseg-containers"
            ) from e


class HDBetWrapper:
    """HD-BET skull stripping wrapper.

    Removes skull from brain MRI scans. Works on T1, T1ce, T2, FLAIR.
    Weights auto-downloaded on first use.
    """

    @staticmethod
    def is_available() -> bool:
        try:
            import hd_bet  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def strip_skull(input_path: str, output_path: str | None = None,
                    device: str = "cpu", disable_tta: bool = False) -> str:
        """Strip skull from brain MRI.

        Args:
            input_path: Path to NIfTI file
            output_path: Path for output (default: <input>_bet.nii.gz)
            device: "cpu" or "cuda"
            disable_tta: Disable test-time augmentation (faster, slightly less accurate)

        Returns:
            Path to skull-stripped output
        """
        if output_path is None:
            output_path = str(Path(input_path).stem) + "_bet.nii.gz"

        try:
            from hd_bet import run as hd_bet_run
            hd_bet_run(input_path, output_path, device=device, disable_tta=disable_tta)
            logger.info("HD-BET skull stripping complete: %s", output_path)
            return str(output_path)
        except ImportError:
            raise ImportError(
                "HD-BET not installed. Install with: pip install brainles-hd-bet"
            )

    @staticmethod
    def strip_skull_batch(input_dir: str, output_dir: str,
                           device: str = "cpu") -> list[str]:
        """Strip skull for all NIfTI files in a directory."""
        from glob import glob
        nifti_files = glob(os.path.join(input_dir, "*.nii*"))
        results = []
        for f in nifti_files:
            out = os.path.join(
                output_dir,
                os.path.basename(f).replace(".nii", "_bet.nii")
            )
            results.append(HDBetWrapper.strip_skull(f, out, device))
        return results


class EDSRPretrained:
    """EDSR super-resolution with pretrained weight download.

    Downloads EDSR weights from a public checkpoint repository.
    The EDSR model was described in "Enhanced Deep Residual Networks
    for Single Image Super-Resolution" (Lim et al., CVPR 2017).
    """

    @staticmethod
    def download_weights(target_dir: str = "./weights") -> str:
        """Download pretrained EDSR weights.

        Args:
            target_dir: Directory to save weights

        Returns:
            Path to downloaded weights file
        """
        import wget
        os.makedirs(target_dir, exist_ok=True)
        weight_path = os.path.join(target_dir, "edsr_x2.pt")

        if os.path.exists(weight_path):
            logger.info("EDSR weights already exist: %s", weight_path)
            return weight_path

        # Try multiple sources
        urls = [
            "https://huggingface.co/pretrained/edsr/resolve/main/edsr_x2.pt",
            "https://github.com/Saige1994/EDSR-TensorFlow/releases/download/v1.0/edsr_x2.pt",
        ]

        for url in urls:
            try:
                logger.info("Downloading EDSR weights from %s ...", url)
                wget.download(url, weight_path)
                logger.info("\nEDSR weights saved to %s", weight_path)
                return weight_path
            except Exception as e:
                logger.warning("Failed to download from %s: %s", url, e)
                continue

        raise RuntimeError(
            "Could not download EDSR weights. Please manually download from "
            "https://huggingface.co/pretrained/edsr"
        )

    @staticmethod
    def is_available() -> bool:
        """Check if EDSR model can be loaded."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False


def download_all_pretrained(target_dir: str = "./pretrained_models") -> dict[str, str]:
    """Download all available pretrained models.

    Returns:
        Dict of model_name -> download path
    """
    results = {}
    try:
        results["edsr"] = EDSRPretrained.download_weights(
            os.path.join(target_dir, "edsr")
        )
    except Exception as e:
        logger.warning("Failed to download EDSR: %s", e)
    logger.info("Downloaded models: %s", list(results.keys()))
    return results
