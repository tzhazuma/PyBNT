"""
Elastix download utility.

Downloads Elastix from GitHub releases for the current platform.
"""
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


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
        print(f"Platform {system} is not supported for elastix.")
        return "elastix"

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
        print(f"Downloaded elastix to {zip_path}")
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
    lib_dir = extract_dir / "lib"

    if str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(lib_dir))

    if system in ("Linux", "Darwin"):
        for exe in ("elastix", "transformix"):
            src = bin_dir / exe
            dst = Path("/usr/local/bin") / exe
            if not src.exists() or dst.exists():
                continue
            try:
                if hasattr(os, "symlink"):
                    dst.symlink_to(src)
                else:
                    shutil.copy2(str(src), str(dst))
            except (PermissionError, OSError):
                print(f"Warning: Cannot install {exe} to /usr/local/bin. "
                      f"Add {bin_dir} to your PATH instead.")

        for lib_name in ("libANNlib-5.1.1.so", "libANNlib-5.1.1.dylib",
                         "libANNlib-5.1.1.1.dylib"):
            src = lib_dir / lib_name
            dst = Path("/usr/local/lib") / lib_name
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(str(src), str(dst))
                except PermissionError:
                    break

    result_bin = shutil.which("elastix") or str(bin_dir / "elastix")
    print(f"Elastix ready at: {result_bin}")
    return result_bin
