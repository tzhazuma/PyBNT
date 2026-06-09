"""FreeSurfer external tool wrapper."""
import os
import platform
import subprocess


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

    setup_cmd = f"source {freesurfer_home}/SetUpFreeSurfer.sh"
    cmd_str = (
        f"{setup_cmd} ; "
        f"export SUBJECTS_DIR={input_path} ; "
        f"export FS_ALLOW_DEEP=1 ; "
        f"recon-all -parallel -i {input_name} -s SPLIT -sd {output_path} -cw256 -all"
    )
    os.system(cmd_str)


def download_freesurfer(target_dir: str = ".") -> str:
    """Download FreeSurfer if not already present."""
    import wget

    freesurfer_dir = os.path.join(target_dir, "freesurfer")
    if os.path.exists(freesurfer_dir):
        return freesurfer_dir

    system = platform.system()
    if system == "Windows":
        print("FreeSurfer doesn't support Windows, please use WSL instead.")
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
        print(f"Unsupported platform: {system}")
        return freesurfer_dir

    tar_path = os.path.join(target_dir, "freesurfer.tar.gz")
    wget.download(url, tar_path)
    os.system(f"tar -xvzf {tar_path} -C {target_dir}")
    os.environ["FREESURFER_HOME"] = freesurfer_dir

    return freesurfer_dir
