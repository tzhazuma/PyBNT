"""
General-purpose utility functions for the PyBNT toolbox.

Includes type-testing helpers, distance/metric computations,
curvature estimation, mesh distance, and image file loading.
"""

from typing import Optional, Union

import cv2
import numpy as np
import pymeshlab  # type: ignore[import-untyped]


def isint(a: str) -> bool:
    """Return ``True`` if *a* can be parsed as an integer."""
    try:
        int(a)
        return True
    except ValueError:
        return False


def isfloat(a: str) -> bool:
    """Return ``True`` if *a* can be parsed as a float."""
    try:
        float(a)
        return True
    except ValueError:
        return False


def dis(v1: Union[np.ndarray, tuple, list], v2: Union[np.ndarray, tuple, list]) -> float:
    """Compute the squared Euclidean distance between two vectors."""
    a1 = np.asarray(v1)
    a2 = np.asarray(v2)
    dif = a2 - a1
    return float(dif.dot(dif))


def gaussian_dis(
    v1: Union[np.ndarray, tuple, list],
    v2: Union[np.ndarray, tuple, list],
    sigma: float = 1.0,
) -> float:
    """Compute a Gaussian-weighted distance between two vectors.

    Returns ``exp(-‖v1-v2‖² / (2σ²))``.
    """
    a1 = np.asarray(v1)
    a2 = np.asarray(v2)
    dif = a2 - a1
    return float(np.exp(-dif.dot(dif) / (2 * (sigma ** 2))))


def vec_len(vec: np.ndarray) -> float:
    """Compute the Euclidean length (L2 norm) of a vector."""
    return float(np.sqrt(vec.dot(vec)))


def angle(vec1: Union[np.ndarray, tuple, list], vec2: Union[np.ndarray, tuple, list]) -> float:
    """Compute the angle (in radians) between two vectors."""
    v1 = np.asarray(vec1)
    v2 = np.asarray(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.arccos(v1.dot(v2) / denom))


def proj(vec1: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Compute the component of *vec1* orthogonal to *direct*."""
    d_len = vec_len(direct)
    if d_len == 0:
        return vec1
    return vec1 - vec1.dot(direct) / d_len * (direct / d_len)


def curvature(
    mesh: Optional[pymeshlab.Mesh] = None,
    path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal curvatures of a mesh using PyMeshLab.

    Args:
        mesh: A PyMeshLab mesh object. If ``None``, *path* must be provided.
        path: Path to a mesh file. Used when *mesh* is ``None``.

    Returns:
        Tuple of ``(mean_curvature, gaussian_curvature, min_curvature)``
        as numpy arrays.
    """
    ms = pymeshlab.MeshSet()
    if path:
        ms.load_new_mesh(path)
    else:
        if mesh is None:
            raise ValueError("Either mesh or path must be provided")
        mesh.save("temp.obj")
        ms.load_new_mesh("temp.obj")

    ms.compute_curvature_principal_directions()

    current_mesh = ms.current_mesh()
    return (
        current_mesh.vertex_mean_curvature(),
        current_mesh.vertex_gaussian_curvature(),
        current_mesh.vertex_min_curvature(),
    )


def disort(
    tri: tuple,
    direct: tuple = (0.0, 0.0, 0.0),
    distortion_type: str = "square",
) -> float:
    """Compute distortion of a triangle under projection along *direct*.

    Args:
        tri: Tuple of three nodes ``(n1, n2, n3)``, each a coordinate tuple.
        direct: Projection direction.
        distortion_type: ``"square"`` for area ratio, ``"angle"`` for angle ratio.

    Returns:
        Distortion ratio (``proj_area / original_area`` or
        ``proj_angle / original_angle``).
    """
    n1, n2, n3 = tri[0:3]
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    n3 = np.asarray(n3)
    d1 = dis(n1, n2)
    d2 = dis(n1, n3)
    v1 = n2 - n1
    v2 = n3 - n1
    sq = 0.5 * d1 * d2 * angle(v1, v2)
    v11 = proj(v1, np.asarray(direct))
    v22 = proj(v2, np.asarray(direct))
    d11 = vec_len(v11)
    d22 = vec_len(v22)
    sq2 = 0.5 * d11 * d22 * angle(v11, v22)
    if distortion_type == "square":
        return float(sq2 / sq)
    elif distortion_type == "angle":
        return float(angle(v11, v22) / angle(v1, v2))
    else:
        print("not support type!")
        return 0.0


def avrdistance(mesh1, mesh2) -> float:
    """Compute the average distance between two point clouds.

    Uses a KD-tree to find the nearest correspondence from each point
    in mesh1 to mesh2 and vice-versa, then returns the mean of all
    those distances.

    Args:
        mesh1: Object with ``.points`` attribute (Nx3 array).
        mesh2: Object with ``.points`` attribute (Mx3 array).

    Returns:
        Mean nearest-neighbour distance.
    """
    from scipy.spatial import KDTree

    points1 = mesh1.points
    points2 = mesh2.points

    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    distances1_to_2, _ = tree2.query(points1)
    distances2_to_1, _ = tree1.query(points2)

    all_distances = np.concatenate([distances1_to_2, distances2_to_1])
    return float(np.mean(all_distances))


def readimage(imagep: str) -> np.ndarray:
    """Load an image file, optionally as NIfTI.

    If the path ends with ``.nii`` or ``.nii.gz`` the image is loaded
    via nibabel and the ``get_fdata()`` array is returned.  Otherwise
    the file is loaded as a regular image with OpenCV.

    Args:
        imagep: Path to the image file.

    Returns:
        Numpy array of the image data.
    """
    if imagep.endswith(".nii") or imagep.endswith(".nii.gz"):
        import nibabel as nib

        img = nib.load(imagep)
        return img.get_fdata()
    else:
        return cv2.imread(imagep)


# Backward-compatible alias for the old "len" name (now vec_len)
len = vec_len
