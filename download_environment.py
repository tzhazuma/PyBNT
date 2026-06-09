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
    print("Installing Python dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    print("All downloads complete!")