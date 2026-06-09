"""
FreeSurfer download utility.

Downloads FreeSurfer from the official Harvard repository.
NOTE: FreeSurfer requires macOS or Linux. Windows users should use WSL.
"""
import os
import platform
import subprocess
from pathlib import Path


def download_freesurfer(target_dir: str | None = None) -> str:
    """Download FreeSurfer to the specified directory.

    Args:
        target_dir: Target directory (default: current working directory).

    Returns:
        Path to the downloaded FreeSurfer directory.
    """
    if target_dir is None:
        target_dir = os.getcwd()

    freesurfer_dir = Path(target_dir) / "freesurfer"
    if freesurfer_dir.exists():
        print(f"FreeSurfer already exists at {freesurfer_dir}")
        os.environ.setdefault("FREESURFER_HOME", str(freesurfer_dir))
        return str(freesurfer_dir)

    system = platform.system()
    if system == "Windows":
        print("FreeSurfer does not support Windows. Please use WSL instead.")
        return str(freesurfer_dir)

    if system == "Linux":
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
        print(f"Unsupported platform: {system}")
        return str(freesurfer_dir)

    tar_path = Path(target_dir) / "freesurfer.tar.gz"
    if not tar_path.exists():
        print(f"Downloading FreeSurfer from {url} ...")
        try:
            subprocess.run(
                ["wget", "--timeout=300", "-O", str(tar_path), url],
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
    else:
        print(f"FreeSurfer archive already exists at {tar_path}")

    print("\nExtracting FreeSurfer...")
    try:
        subprocess.run(["tar", "-xvzf", str(tar_path), "-C", target_dir],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract FreeSurfer: {e.stderr.decode() if e.stderr else str(e)}"
        )

    os.environ.setdefault("FREESURFER_HOME", str(freesurfer_dir))
    print(f"FreeSurfer extracted to {freesurfer_dir}")
    print("NOTE: Run 'source $FREESURFER_HOME/SetUpFreeSurfer.sh' to configure.")
    return str(freesurfer_dir)
