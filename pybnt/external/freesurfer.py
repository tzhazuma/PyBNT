"""FreeSurfer external tool wrapper."""
import os
import platform
import subprocess

from pybnt.core.logconf import logger


def run_freesurfer(input_path: str, subject_id: str, output_dir: str,
                   freesurfer_home: str | None = None):
    """Run FreeSurfer recon-all pipeline.

    Requires FreeSurfer to be installed. Falls back to downloading if not found.
    """
    if freesurfer_home is None:
        freesurfer_home = os.environ.get("FREESURFER_HOME", "")
    if not freesurfer_home or not os.path.exists(freesurfer_home):
        freesurfer_home = download_freesurfer()

    cmd = [
        os.path.join(freesurfer_home, "bin", "recon-all"),
        "-s", subject_id,
        "-i", input_path,
        "-all"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def split_freesurfer(input_path: str, input_name: str, output_path: str):
    """Split FreeSurfer output data.

    Runs recon-all with -parallel and -cw256 flags for hemisphere splitting.
    """
    freesurfer_home = os.environ.get(
        "FREESURFER_HOME", os.path.join(os.getcwd(), "freesurfer")
    )
    if not os.path.exists(freesurfer_home):
        download_freesurfer()
        freesurfer_home = os.environ.get(
            "FREESURFER_HOME", os.path.join(os.getcwd(), "freesurfer")
        )

    setup_sh = os.path.join(freesurfer_home, "SetUpFreeSurfer.sh")
    env = os.environ.copy()
    env["SUBJECTS_DIR"] = input_path
    env["FS_ALLOW_DEEP"] = "1"
    cmd = [
        "bash", "-c",
        f"source {setup_sh} && recon-all -parallel "
        f"-i {input_name} -s SPLIT -sd {output_path} -cw256 -all"
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"FreeSurfer split failed: {result.stderr}")


def download_freesurfer(target_dir: str = ".") -> str:
    """Download FreeSurfer if not already present."""

    freesurfer_dir = os.path.join(target_dir, "freesurfer")
    if os.path.exists(freesurfer_dir):
        return freesurfer_dir

    system = platform.system()
    if system == "Windows":
        logger.warning("FreeSurfer doesn't support Windows, please use WSL instead.")
        return freesurfer_dir
    elif system == "Linux":
        url = ("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
               "8.0.0-beta/freesurfer-linux-ubuntu22_x86_64-8.0.0-beta.tar.gz")
    elif system == "Darwin":
        if platform.machine() == "x86_64":
            url = ("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
                   "8.0.0-beta/freesurfer-macOS-darwin_x86_64-8.0.0-beta.tar.gz")
        else:
            url = ("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/"
                   "8.0.0-beta/freesurfer-macOS-darwin_arm64-8.0.0-beta.tar.gz")
    else:
        logger.warning(f"Unsupported platform: {system}")
        return freesurfer_dir

    tar_path = os.path.join(target_dir, "freesurfer.tar.gz")
    try:
        subprocess.run(
            ["wget", "--timeout=300", "-O", tar_path, url],
            check=True, capture_output=True
        )
    except FileNotFoundError:
        raise RuntimeError(
            "wget is not installed. Install it with: apt-get install wget (Linux) "
            "or brew install wget (macOS)"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download FreeSurfer: {e.stderr.decode() if e.stderr else str(e)}"
        )
    try:
        subprocess.run(
            ["tar", "-xvzf", tar_path, "-C", target_dir],
            check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract FreeSurfer: {e.stderr.decode() if e.stderr else str(e)}"
        )
    os.environ["FREESURFER_HOME"] = freesurfer_dir

    return freesurfer_dir
