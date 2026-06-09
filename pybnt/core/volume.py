"""Volume-to-surface mapping algorithms for brain visualization.

Provides functions to map 3D volume data (NIfTI) onto brain surface meshes (.nv).
Compatible with BrainNet Viewer volume mapping format.
"""

import numpy as np


def map_volume_to_surface(
    volume_path: str,
    surface_path: str,
    algorithm: str = "nearest",
    value_range: str = "both",
    radius: float = 5.0,
) -> np.ndarray:
    """Map volume data to surface vertices using specified algorithm.

    Args:
        volume_path: Path to NIfTI volume (.nii/.nii.gz) or text file
        surface_path: Path to brain surface (.nv/.pial)
        algorithm: Mapping algorithm - one of:

            * ``"nearest"`` - Nearest neighbor interpolation
            * ``"linear"`` - Linear interpolation
            * ``"gaussian"`` - Gaussian weighted average
            * ``"cubic"`` - Cubic interpolation
            * ``"maximum"`` - Maximum value within radius
            * ``"minimum"`` - Minimum value within radius
            * ``"average"`` - Average value within radius

        value_range: ``"positive"``, ``"negative"``, or ``"both"``
        radius: Search radius for interpolation (mm)

    Returns:
        Vertex-wise scalar array with same length as surface vertices

    Raises:
        ValueError: If algorithm is unknown or surface/volume loading fails
    """
    from pybnt.core.surface import parse_surface
    from pybnt.core.utils import readimage

    # Load volume data
    volume_data = readimage(volume_path)
    if volume_data is None or not isinstance(volume_data, np.ndarray):
        raise ValueError(f"Failed to load volume: {volume_path}")

    # Load surface mesh
    nodes, _tris = parse_surface(surface_path)
    vertices = np.array(nodes)

    # Determine grid coordinates for volume
    shape = np.array(volume_data.shape)
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    # Apply value range filter
    if value_range == "positive":
        volume_data = np.clip(volume_data, 0, None)
    elif value_range == "negative":
        volume_data = np.clip(volume_data, None, 0)

    # Select and apply mapping algorithm
    algorithms = {
        "nearest": _map_nearest_neighbor,
        "linear": _map_linear_interpolation,
        "gaussian": _map_gaussian_weighted,
        "cubic": _map_cubic_interpolation,
        "maximum": _map_maximum,
        "minimum": _map_minimum,
        "average": _map_average,
    }

    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Choose from: {list(algorithms.keys())}"
        )

    mapper = algorithms[algorithm]
    vertex_values = mapper(volume_data, vertices, grid_x, grid_y, grid_z, radius)

    # Handle NaN values
    vertex_values = np.nan_to_num(vertex_values, nan=0.0)

    return vertex_values


def _map_nearest_neighbor(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using nearest neighbor interpolation."""
    values = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        diffs = np.sqrt(
            (gx - vertex[0]) ** 2
            + (gy - vertex[1]) ** 2
            + (gz - vertex[2]) ** 2
        )
        min_idx = np.unravel_index(np.argmin(diffs), diffs.shape)
        values[i] = volume[min_idx]
    return values


def _map_linear_interpolation(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using linear (trilinear) interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    shape = volume.shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])

    interpolator = RegularGridInterpolator(
        (x, y, z),
        volume,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    values = interpolator(vertices)
    return np.asarray(values)


def _map_gaussian_weighted(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using Gaussian-weighted average of nearby voxels."""
    sigma = radius / 2.0
    values = np.zeros(len(vertices))

    for i, vertex in enumerate(vertices):
        diffs_sq = (
            (gx - vertex[0]) ** 2
            + (gy - vertex[1]) ** 2
            + (gz - vertex[2]) ** 2
        )
        weights = np.exp(-diffs_sq / (2 * sigma**2))
        mask = diffs_sq < radius**2
        if np.any(mask):
            values[i] = np.average(volume[mask], weights=weights[mask])

    return values


def _map_cubic_interpolation(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using cubic interpolation."""
    from scipy.interpolate import RegularGridInterpolator

    shape = volume.shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])

    interpolator = RegularGridInterpolator(
        (x, y, z),
        volume,
        method="cubic",
        bounds_error=False,
        fill_value=0,
    )
    values = interpolator(vertices)
    return np.asarray(values)


def _map_maximum(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using maximum value within search radius."""
    values = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        diffs_sq = (
            (gx - vertex[0]) ** 2
            + (gy - vertex[1]) ** 2
            + (gz - vertex[2]) ** 2
        )
        mask = diffs_sq < radius**2
        if np.any(mask):
            values[i] = np.max(volume[mask])
    return values


def _map_minimum(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using minimum value within search radius."""
    values = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        diffs_sq = (
            (gx - vertex[0]) ** 2
            + (gy - vertex[1]) ** 2
            + (gz - vertex[2]) ** 2
        )
        mask = diffs_sq < radius**2
        if np.any(mask):
            values[i] = np.min(volume[mask])
    return values


def _map_average(
    volume: np.ndarray,
    vertices: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    radius: float = 5.0,
) -> np.ndarray:
    """Map using average value within search radius."""
    values = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        diffs_sq = (
            (gx - vertex[0]) ** 2
            + (gy - vertex[1]) ** 2
            + (gz - vertex[2]) ** 2
        )
        mask = diffs_sq < radius**2
        if np.any(mask):
            values[i] = np.mean(volume[mask])
    return values
