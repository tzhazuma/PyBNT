"""Elastix image registration wrapper."""
import subprocess
import os
import shutil


def register_elastix(fixed_image: str, moving_image: str, output_dir: str,
                     parameter_file: str | None = None,
                     elastix_path: str | None = None) -> str:
    """Register images using elastix.

    Returns path to result image, or empty string on failure.
    """
    if elastix_path is None:
        elastix_path = _find_elastix()
    cmd = [elastix_path, "-f", fixed_image, "-m", moving_image, "-out", output_dir]
    if parameter_file:
        cmd += ["-p", parameter_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return os.path.join(output_dir, "result.0.nii")
    return ""


def _find_elastix() -> str:
    """Find elastix binary in common locations."""
    candidates = [
        "elastix",
        "/usr/local/bin/elastix",
        os.path.join(os.path.dirname(__file__), "..", "..", "correct_elastix", "elastix", "bin", "elastix"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        found = shutil.which(c)
        if found is not None:
            return found
    return "elastix"
