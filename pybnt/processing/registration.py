"""Image registration using elastix."""
import os
import subprocess
import sys


def _get_elastix_path() -> str:
    """Get the path to the elastix binary directory.

    Returns
    -------
    str
        Path to the elastix installation directory.
    """
    if "ELASTIX_PATH" in os.environ:
        return os.environ["ELASTIX_PATH"]
    # Default to looking in the project root
    cwd = os.getcwd()
    elastix_dir = os.path.join(cwd, "elastix")
    if os.path.isdir(elastix_dir):
        return elastix_dir
    raise RuntimeError(
        "elastix not found. Set ELASTIX_PATH environment variable "
        "or download elastix to the current directory."
    )


def _setup_elastix_env():
    """Ensure elastix is available, downloading if necessary."""
    elastix_path = os.path.join(os.getcwd(), "elastix")
    if os.path.isdir(elastix_path):
        os.environ["ELASTIX_PATH"] = elastix_path
        os.environ["PATH"] = (
            os.path.join(elastix_path, "bin") + os.pathsep + os.environ.get("PATH", "")
        )
        os.environ["PATH"] = (
            os.path.join(elastix_path, "lib") + os.pathsep + os.environ["PATH"]
        )
        return

    from pybnt.processing._elastix_download import download_elastix
    download_elastix()
    os.environ["ELASTIX_PATH"] = os.path.join(os.getcwd(), "elastix")
    os.environ["PATH"] = (
        os.path.join(os.getcwd(), "elastix/bin") + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ["PATH"] = (
        os.path.join(os.getcwd(), "elastix/lib") + os.pathsep + os.environ["PATH"]
    )


def register_image(
    fixed_path: str,
    moving_path: str,
    output_path: str | None = None,
) -> str:
    """Register moving image to fixed image using elastix.

    Parameters
    ----------
    fixed_path : str
        Path to the fixed (reference) image.
    moving_path : str
        Path to the moving image to be registered.
    output_path : str or None
        Output directory. If None, uses a subdirectory next to the moving image.

    Returns
    -------
    str
        Path to the registered output image.
    """
    _setup_elastix_env()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(moving_path),
            "registered_output",
        )

    print("elastix registration started ...")
    try:
        subprocess.run(
            [
                "elastix",
                "-f", fixed_path,
                "-m", moving_path,
                "-out", output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"elastix registration failed: {e.stderr}"
        ) from e

    result_path = os.path.join(output_path, "result.0.nii")
    if not os.path.exists(result_path):
        # Try alternative output naming
        result_files = [
            f for f in os.listdir(output_path)
            if f.startswith("result")
        ]
        if result_files:
            result_path = os.path.join(output_path, result_files[0])
        else:
            raise RuntimeError(
                f"No result image found in {output_path}"
            )

    print(f"Registration complete. Result: {result_path}")
    return result_path


def correct_motion(
    image_sequence_dir: str,
    reference_index: int = 0,
    output_dir: str | None = None,
):
    """Motion correction for fMRI time series.

    Registers each frame in a time series to a reference frame.

    Parameters
    ----------
    image_sequence_dir : str
        Directory containing sequential image frames.
    reference_index : int
        Index of the reference frame (0-based).
    output_dir : str or None
        Output directory for corrected images.
        If None, creates a subdirectory.
    """
    import nibabel as nib

    _setup_elastix_env()

    if output_dir is None:
        output_dir = os.path.join(image_sequence_dir, "motion_corrected")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(image_sequence_dir)
        if f.endswith(('.nii', '.nii.gz', '.png', '.jpg', '.dcm'))
    ])

    if not files:
        raise ValueError(f"No image files found in {image_sequence_dir}")

    ref_file = files[reference_index]
    ref_path = os.path.join(image_sequence_dir, ref_file)

    results = []
    for i, filename in enumerate(files):
        if i == reference_index:
            continue
        moving_path = os.path.join(image_sequence_dir, filename)
        frame_output = os.path.join(output_dir, f"corrected_{i:04d}")
        os.makedirs(frame_output, exist_ok=True)

        try:
            subprocess.run(
                [
                    "elastix",
                    "-f", ref_path,
                    "-m", moving_path,
                    "-out", frame_output,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(frame_output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Motion correction failed for frame {i}: {e.stderr}"
            ) from e

    return results


def _need_download() -> bool:
    """Check if elastix needs to be downloaded."""
    return "elastix" not in os.listdir(os.getcwd())
