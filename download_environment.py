"""Download all external tools required by PyBrainViewer."""
import os
import subprocess
import sys

from pybnt.external.freesurfer import download_freesurfer
from pybnt.external.elastix import download_elastix


def download_all():
    """Download FreeSurfer, Elastix, and install Python dependencies."""
    print("Downloading FreeSurfer...")
    download_freesurfer()
    print("Downloading Elastix...")
    download_elastix()
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"Installing Python dependencies from {req_file}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", req_file]
        )
    else:
        print(f"{req_file} not found, skipping dependency installation.")
    print("All downloads complete!")