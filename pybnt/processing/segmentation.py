"""Brain segmentation using FreeSurfer."""
import os
import platform
import subprocess

from pybnt.core.logconf import logger


def _get_freesurfer_home() -> str:
    """Get the FreeSurfer installation directory.

    Returns
    -------
    str
        Path to FreeSurfer home directory.
    """
    if "FREESURFER_HOME" in os.environ:
        return os.environ["FREESURFER_HOME"]
    # Check current directory
    freesurfer_dir = os.path.join(os.getcwd(), "freesurfer")
    if os.path.isdir(freesurfer_dir):
        return freesurfer_dir
    if platform.system() == "Darwin":
        default_path = "/Applications/freesurfer"
    else:
        default_path = "/usr/local/freesurfer"
    if os.path.isdir(default_path):
        return default_path
    raise RuntimeError(
        "FreeSurfer not found. Set FREESURFER_HOME or run "
        "download_freesurfer() first."
    )


def _set_freesurfer_env():
    """Configure FreeSurfer environment variables."""
    fshome = _get_freesurfer_home()
    os.environ["FREESURFER_HOME"] = fshome


def segment_brain(input_path: str, subject_id: str, output_dir: str):
    """Run FreeSurfer recon-all pipeline for brain segmentation.

    Parameters
    ----------
    input_path : str
        Path to input T1-weighted MRI image.
    subject_id : str
        Subject identifier for FreeSurfer output directory.
    output_dir : str
        Directory where subject output will be stored (SUBJECTS_DIR).

    Returns
    -------
    int
        Return code from recon-all.
    """
    _set_freesurfer_env()
    fshome = os.environ["FREESURFER_HOME"]

    if not os.path.isdir(fshome):
        raise FileNotFoundError(f"FreeSurfer not found at {fshome}")

    cmd = (
        f"source {fshome}/SetUpFreeSurfer.sh && "
        f"export SUBJECTS_DIR={output_dir} && "
        f"export FS_ALLOW_DEEP=1 && "
        f"recon-all -parallel -i {input_path} -s {subject_id} "
        f"-sd {output_dir} -cw256 -all"
    )

    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"recon-all failed (code {result.returncode}):\n{result.stderr}"
        )
    return result.returncode


def split_freesurfer(input_path: str, input_name: str, output_path: str):
    """Split FreeSurfer output by hemisphere.

    Convenience wrapper that runs recon-all on a subject.

    Parameters
    ----------
    input_path : str
        Path containing input data.
    input_name : str
        Name of the input image file.
    output_path : str
        Output directory for split results.
    """
    _set_freesurfer_env()
    fshome = os.environ["FREESURFER_HOME"]

    cmd = (
        f"source {fshome}/SetUpFreeSurfer.sh && "
        f"export SUBJECTS_DIR={input_path} && "
        "export FS_ALLOW_DEEP=1 && "
        f"recon-all -parallel -i {input_name} -s SPLIT "
        f"-sd {output_path} -cw256 -all"
    )

    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FreeSurfer split failed (code {result.returncode}):\n{result.stderr}"
        )


def download_freesurfer(target_dir: str = "freesurfer"):
    """Download FreeSurfer if not installed.

    Parameters
    ----------
    target_dir : str
        Directory to extract FreeSurfer into.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Windows":
        logger.warning("FreeSurfer does not support Windows. Use WSL instead.")
        return

    import wget

    if system == "Linux":
        url = (
            "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
            "8.0.0-beta/freesurfer-linux-ubuntu22_x86_64-8.0.0-beta.tar.gz"
        )
    elif system == "Darwin":
        if machine == "x86_64":
            url = (
                "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
                "8.0.0-beta/freesurfer-macOS-darwin_x86_64-8.0.0-beta.tar.gz"
            )
        else:
            url = (
                "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
                "8.0.0-beta/freesurfer-macOS-darwin_arm64-8.0.0-beta.tar.gz"
            )
    else:
        logger.warning(f"Unsupported platform: {system}")
        return

    logger.info(f"Downloading FreeSurfer from {url} ...")
    wget.download(url, "freesurfer.tar.gz")

    subprocess.run(["tar", "-xvzf", "freesurfer.tar.gz"], check=True)
    subprocess.run(
        f"chmod -R 777 ./{target_dir}", shell=True, check=True
    )
    os.environ["FREESURFER_HOME"] = os.path.join(os.getcwd(), target_dir)


def parse_pial_surface(path: str):
    """Parse a FreeSurfer pial surface file.

    Parameters
    ----------
    path : str
        Path to the surface geometry file.

    Returns
    -------
    tuple
        (nodes, edges) arrays from the surface.
    """
    import nibabel as nib
    nodes, edges = nib.freesurfer.io.read_geometry(path)
    return nodes, edges
