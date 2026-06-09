"""SPM12 external tool wrapper.

Provides Pythonic access to SPM12 brain imaging operations with
automatic fallback to nibabel for basic NIfTI read/write when
the ``spm12`` MATLAB bridge is unavailable.

MATLAB Dependency
-----------------
Full SPM12 functionality (batch processing, model estimation)
requires a MATLAB Runtime (MCR) installation and the ``spm12``
Python package (``pip install spm12``). Without these, only
nibabel-based NIfTI I/O and pure-Python GLM estimation are
available.

Example
-------
>>> from pybnt.external.spm import read_spm_volume, write_spm_volume
>>> vol = read_spm_volume("test.nii")       # nibabel fallback
>>> result = glm_analysis(X, y)             # pure-Python OLS
"""

from __future__ import annotations

import os
import subprocess
import warnings
from typing import Any

import numpy as np

from pybnt.core.logconf import logger

__all__ = [
    "read_spm_volume",
    "write_spm_volume",
    "parse_spm_mat",
    "run_spm_batch",
    "glm_analysis",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_spm12() -> Any:
    """Import and return the ``spm12`` module, or ``None`` if unavailable.

    Returns
    -------
    module or None
        The ``spm12`` module when MATLAB/MCR is available, else ``None``.
    """
    try:
        import spm12 as _spm
        return _spm
    except ImportError:
        warnings.warn(
            "spm12 package not available; install with ``pip install spm12`` "
            "(requires MATLAB Runtime). Falling back to nibabel for I/O.",
            stacklevel=2,
        )
        return None


def _get_nibabel() -> Any:
    """Import and return ``nibabel``, raising on failure."""
    try:
        import nibabel as nib
        return nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for NIfTI operations. "
            "Install with ``pip install nibabel``."
        ) from exc


# ---------------------------------------------------------------------------
# NIfTI volume I/O
# ---------------------------------------------------------------------------

def read_spm_volume(path: str) -> np.ndarray:
    """Read a NIfTI volume from disk.

    Tries the SPM12 MATLAB bridge first; falls back to nibabel if the
    ``spm12`` package is not installed.

    Parameters
    ----------
    path : str
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    np.ndarray
        The image data as a multi-dimensional NumPy array (typically
        3-D for structural or 4-D for fMRI).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If neither ``spm12`` nor ``nibabel`` is available.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    spm = _get_spm12()
    if spm is not None:
        try:
            vol = spm.spm_vol(path)
            data = spm.spm_read_vols(vol)
            return np.asarray(data)
        except Exception as exc:
            warnings.warn(
                f"SPM12 read failed ({exc}); falling back to nibabel.",
                stacklevel=2,
            )

    nib = _get_nibabel()
    img = nib.load(path)
    return np.asarray(img.get_fdata())


def write_spm_volume(
    data: np.ndarray,
    path: str,
    affine: np.ndarray | None = None,
) -> None:
    """Write a NumPy array as a NIfTI volume.

    This function always uses nibabel (pure Python, no MATLAB required).

    Parameters
    ----------
    data : np.ndarray
        3-D or 4-D image data.
    path : str
        Output path (``.nii`` or ``.nii.gz``).
    affine : np.ndarray, optional
        4x4 affine transformation matrix mapping voxel indices to
        world coordinates. Defaults to identity.
    """
    nib = _get_nibabel()
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


# ---------------------------------------------------------------------------
# SPM.mat parsing
# ---------------------------------------------------------------------------

def parse_spm_mat(path: str) -> dict:
    """Parse an ``SPM.mat`` file and return its contents as a dictionary.

    SPM.mat is a MATLAB ``.mat`` file created by SPM12 during model
    specification. This function extracts the ``SPM`` structure and
    flattens commonly-used fields for easier access.

    Parameters
    ----------
    path : str
        Path to an ``SPM.mat`` file.

    Returns
    -------
    dict
        Parsed contents. If the file cannot be read (missing, corrupt,
        or ``scipy`` unavailable), an empty dictionary is returned and
        a warning is emitted.

    Note
    ----
    Requires ``scipy`` (``pip install scipy``).
    """
    if not os.path.isfile(path):
        warnings.warn(f"SPM.mat file not found: {path}", stacklevel=2)
        return {}

    try:
        import scipy.io as sio
    except ImportError:
        warnings.warn(
            "scipy is required to parse SPM.mat files. "
            "Install with ``pip install scipy``.",
            stacklevel=2,
        )
        return {}

    try:
        raw: dict = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

        # If the expected ``SPM`` variable exists, return it keyed by field
        spm_struct = raw.get("SPM")
        if spm_struct is not None:
            # flattens the MATLAB struct into a readable dict
            out: dict[str, Any] = {}
            for field_name in spm_struct._fieldnames:
                out[field_name] = getattr(spm_struct, field_name)
            return out

        return raw

    except Exception as exc:
        warnings.warn(f"Error parsing SPM.mat: {exc}", stacklevel=2)
        return {}


# ---------------------------------------------------------------------------
# Running SPM batch scripts
# ---------------------------------------------------------------------------

def _find_matlab() -> str | None:
    """Locate a MATLAB or MCR executable on ``$PATH``.

    Returns
    -------
    str or None
        Absolute path to the ``matlab`` binary, or ``None`` if not found.
    """
    import shutil
    return shutil.which("matlab")


def run_spm_batch(
    script_path: str,
    matlab_cmd: str | None = None,
    timeout: int | None = None,
) -> bool:
    """Execute an SPM batch job via MATLAB.

    Parameters
    ----------
    script_path : str
        Path to a MATLAB ``.m`` script that contains an SPM batch
        invocation (e.g., ``spm_jobman('run', matlabbatch)``).
    matlab_cmd : str, optional
        The MATLAB executable or command. Defaults to ``matlab``
        (found via ``$PATH``).
    timeout : int, optional
        Maximum wall-clock time in seconds before the process is
        killed. ``None`` means no limit.

    Returns
    -------
    bool
        ``True`` if the batch script completed successfully (exit
        code 0), ``False`` otherwise.

    Raises
    ------
    FileNotFoundError
        If *script_path* does not exist.
    RuntimeError
        If MATLAB is not installed or not on ``$PATH``.

    Note
    ----
    Requires a full MATLAB installation (not just MCR) or an MCR
    with a compatible ``matlab`` executable wrapper. The SPM12
    toolbox must be on the MATLAB path.
    """
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Batch script not found: {script_path}")

    if matlab_cmd is None:
        found = _find_matlab()
        if found is None:
            raise RuntimeError(
                "MATLAB executable not found on $PATH. "
                "Install MATLAB or provide the path via ``matlab_cmd``."
            )
        matlab_cmd = found

    try:
        result = subprocess.run(
            [matlab_cmd, "-batch", f"run('{script_path}')"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.error("MATLAB stderr: %s", result.stderr)
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        warnings.warn(
            f"MATLAB batch timed out after {timeout}s.", stacklevel=2
        )
        return False
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"MATLAB executable '{matlab_cmd}' not found."
        ) from exc


# ---------------------------------------------------------------------------
# General Linear Model (pure Python)
# ---------------------------------------------------------------------------

def glm_analysis(
    design_matrix: np.ndarray,
    data: np.ndarray,
) -> dict[str, Any]:
    """Fit a General Linear Model using ordinary least squares.

    This is a pure-Python/NumPy implementation that does **not** require
    MATLAB or SPM12.

    Parameters
    ----------
    design_matrix : np.ndarray, shape (N, P)
        Design matrix where rows = observations and columns = regressors.
    data : np.ndarray, shape (N,) or (N, ...)
        Observed data. If multi-dimensional, the first axis must match
        *design_matrix* and the remaining dimensions are treated as
        independent regression problems (e.g., voxels).

    Returns
    -------
    dict
        Dictionary with keys:

        - **betas** (*np.ndarray*) – estimated regression coefficients
          shape ``(P,)`` or ``(P, ...)``.
        - **residuals** (*np.ndarray*) – residuals ``y - X @ beta``,
          same shape as *data*.
        - **rss** (*np.ndarray*) – residual sum of squares, shape
          ``(...)`` (squeezed to match trailing dims of *data*).
        - **dof** (*int*) – residual degrees of freedom ``N - P``.
        - **mse** (*np.ndarray*) – mean squared error (RSS / dof),
          shape ``(...)``.
        - **cov_beta** (*np.ndarray*) – covariance matrix of beta
          estimates, shape ``(P, P)`` (only for 1-D *data*; ``None``
          otherwise).
        - **t_stat** (*np.ndarray*) – t-statistics for each
          coefficient, shape ``(P,)`` or ``(P, ...)``.
        - **r_squared** (*float*) – coefficient of determination
          (only for 1-D *data*; ``None`` otherwise).

    Raises
    ------
    ValueError
        If the design matrix is singular or dimensions are
        incompatible.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = X @ np.array([1.0, -0.5, 0.2]) + 0.1 * np.random.randn(100)
    >>> result = glm_analysis(X, y)
    >>> result["betas"].shape
    (3,)
    """
    X = np.asarray(design_matrix, dtype=np.float64)
    y = np.asarray(data, dtype=np.float64)

    N, P = X.shape

    if y.shape[0] != N:
        raise ValueError(
            f"First dimension of data ({y.shape[0]}) must match "
            f"number of rows in design matrix ({N})."
        )

    # Check rank
    rank = np.linalg.matrix_rank(X)
    if rank < P:
        raise ValueError(
            f"Design matrix is rank-deficient (rank {rank} < {P} columns)."
        )

    # --- OLS estimation ---
    # Pinv is numerically stable even for ill-conditioned X
    X_pinv = np.linalg.pinv(X)
    orig_shape = y.shape

    # Flatten trailing dimensions for vectorisation
    if y.ndim == 1:
        y_2d = y[:, np.newaxis]       # (N, 1)
    else:
        y_2d = y.reshape(N, -1)       # (N, V)

    betas_2d = X_pinv @ y_2d          # (P, V)
    residuals_2d = y_2d - X @ betas_2d  # (N, V)
    rss_2d = np.sum(residuals_2d ** 2, axis=0)  # (V,)
    dof = N - P
    mse_2d = rss_2d / dof if dof > 0 else np.full_like(rss_2d, np.nan)

    # t-statistics
    xtx_inv = np.linalg.inv(X.T @ X)
    se_2d = np.sqrt(mse_2d * np.diag(xtx_inv)[:, np.newaxis])  # (P, V)
    t_stat_2d = betas_2d / se_2d

    # Restore shapes
    if y.ndim == 1:
        betas = betas_2d[:, 0]
        residuals = residuals_2d[:, 0]
        rss = rss_2d[0]
        mse = mse_2d[0]
        t_stat = t_stat_2d[:, 0]
        cov_beta = mse * xtx_inv
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - rss / ss_total if ss_total > 0 else 0.0
    else:
        betas = betas_2d.reshape(P, *orig_shape[1:])
        residuals = residuals_2d.reshape(orig_shape)
        rss = rss_2d.reshape(orig_shape[1:])
        mse = mse_2d.reshape(orig_shape[1:])
        t_stat = t_stat_2d.reshape(P, *orig_shape[1:])
        cov_beta = None
        r_squared = None

    return {
        "betas": betas,
        "residuals": residuals,
        "rss": rss,
        "dof": dof,
        "mse": mse,
        "cov_beta": cov_beta,
        "t_stat": t_stat,
        "r_squared": r_squared,
    }
