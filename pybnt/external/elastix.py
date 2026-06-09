"""Elastix image registration wrapper."""
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path


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
        os.path.join(os.path.dirname(__file__), "..", "..", "correct_elastix",
                     "elastix", "bin", "elastix"),
        os.path.join(os.path.dirname(__file__), "..", "..", "elastix", "bin",
                     "elastix"),
        os.path.expanduser("~/elastix/bin/elastix"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        found = shutil.which(c)
        if found is not None:
            return found
    return "elastix"


def download_elastix(target_dir: str | None = None) -> str:
    """Download Elastix binaries for the current platform.

    Downloads from GitHub releases (5.1.0) and extracts to target_dir/elastix/.

    Args:
        target_dir: Target directory (default: current working directory).

    Returns:
        Path to the elastix binary.
    """
    if target_dir is None:
        target_dir = os.getcwd()

    system = platform.system()
    platform_suffixes = {
        "Windows": "win64",
        "Linux": "linux",
        "Darwin": "mac",
    }

    suffix = platform_suffixes.get(system)
    if suffix is None:
        raise RuntimeError(f"Platform {system} is not supported for elastix.")

    url = (f"https://github.com/SuperElastix/elastix/releases/download/5.1.0/"
           f"elastix-5.1.0-{suffix}.zip")
    zip_name = f"elastix-5.1.0-{suffix}.zip"
    zip_path = Path(target_dir) / zip_name

    if not zip_path.exists():
        print(f"Downloading elastix from {url} ...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                subprocess.run(
                    ["wget", "--timeout=120", "-O", str(zip_path), url],
                    check=True, capture_output=True
                )
                break
            except FileNotFoundError:
                raise RuntimeError(
                    "wget is not installed. Install it with: apt-get install wget (Linux) "
                    "or brew install wget (macOS)"
                )
            except subprocess.CalledProcessError as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download elastix after {max_retries} attempts: "
                        f"{e.stderr.decode() if e.stderr else str(e)}"
                    )
                print(f"Download attempt {attempt} failed, retrying...")
    else:
        print(f"Elastix archive already exists at {zip_path}")

    extract_dir = Path(target_dir) / "elastix"
    if not extract_dir.exists():
        print(f"\nExtracting to {extract_dir} ...")
        try:
            with zipfile.ZipFile(str(zip_path)) as zf:
                zf.extractall(str(extract_dir))
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Downloaded file is not a valid zip file: {e}")
    else:
        print(f"Already extracted at {extract_dir}")

    bin_dir = extract_dir / "bin"
    if str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    result_bin = shutil.which("elastix") or str(bin_dir / "elastix")
    print(f"Elastix ready at: {result_bin}")
    return result_bin