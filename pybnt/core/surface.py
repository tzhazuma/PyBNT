"""
Surface file parsing module.

Supports the following surface formats:

- .nv (native): Custom text format used by BrainNet Viewer.
  The file contains a node count on one line, followed by that many
  lines of space-separated ``x y z`` coordinates, then a triangle
  count on one line, followed by that many lines of space-separated
  ``v1 v2 v3`` triangle indices.

- .pial: FreeSurfer pial surface, parsed via ``free_surfer.parse``.

- .gii: GIFTI surface format, loaded via ``nibabel``.

- .obj: Wavefront OBJ format with ``v`` and ``f`` lines.

- .mz3: Surf Ice MZ3 format, loaded via ``nibabel``.

All parsers return ``(nodes, triangles)`` as ``(N, 3)`` and ``(M, 3)``
numpy arrays respectively.
"""

import os

import numpy as np

from pybnt.core.utils import isint


def parse_surface(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a brain surface file, auto-detecting format from extension.

    Returns a tuple of ``(nodes, triangles)`` where ``nodes`` is an
    ``(N, 3)`` numpy array of vertex coordinates and ``triangles`` is
    an ``(M, 3)`` numpy array of integer vertex indices.

    Supported formats: ``.nv``, ``.pial``, ``.gii``, ``.obj``, ``.mz3``.
    """
    if path.endswith(".pial"):
        import free_surfer.parse as parsepial  # type: ignore[import-untyped]

        nodes, tris = parsepial.parse_pial(path)
        return np.array(nodes), np.array(tris)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".gii":
        return _parse_gii(path)

    if ext == ".obj":
        return _parse_obj(path)

    if ext == ".mz3":
        return _parse_mz3(path)

    # Default: parse .nv (BrainNet Viewer native format)
    return _parse_nv(path)


def _parse_nv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a native ``.nv`` surface file.

    File structure::

        <nodenum>
        x1 y1 z1
        ...
        <trinum>
        v1a v1b v1c
        ...
    """
    with open(path, "r") as f:
        nodenum = 0
        nodes: list[list[float]] = []

        # Skip non-integer lines until the node count is found
        while True:
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading node count from {path}"
                )
            if not isint(line):
                continue
            nodenum = int(line)
            break

        print(f"Parsing Nodenums: {nodenum}")
        print("Starting parsing nodes position")

        for i in range(nodenum):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF at node {i}")
            parts = line.strip().split(" ")
            if len(parts) < 3:
                raise ValueError(
                    f"Nodes data error at node {i}: expected x y z"
                )
            x, y, z = parts[0:3]
            nodes.append([float(x), float(y), float(z)])

        trinum_line = f.readline()
        if not trinum_line:
            raise ValueError("Unexpected EOF while reading triangle count")
        if not isint(trinum_line):
            raise ValueError(
                f"Error parsing triangle count: {trinum_line.strip()!r}"
            )

        triangles: list[list[int]] = []
        for i in range(int(trinum_line)):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF at triangle {i}")
            parts = line.strip().split(" ")
            if len(parts) < 3:
                raise ValueError(
                    f"Triangle data error at triangle {i}: expected v1 v2 v3"
                )
            x, y, z = parts[0:3]
            triangles.append([int(x), int(y), int(z)])

    return np.array(nodes), np.array(triangles)


def _parse_gii(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a GIFTI (``.gii``) surface file using nibabel.

    Expects the first data array to contain vertex coordinates and the
    second data array to contain triangle indices.
    """
    import nibabel as nib

    gii_img = nib.load(path)
    darrays = gii_img.darrays

    # Try to identify vertices and triangles by dimensions/intent
    nodes = None
    tris = None
    for darray in darrays:
        if darray.dims[0] == 3 and darray.dims[1] > 3:
            # Vertices: shape (N, 3) stored as (3, N) in dims
            nodes = np.array(darray.data)
        elif hasattr(darray, "intent") and hasattr(
            nib.nifti2, "NIFTI_INTENT_TRIANGLE"
        ):
            if darray.intent == nib.nifti2.NIFTI_INTENT_TRIANGLE:
                tris = np.array(darray.data)

    # Fallback: assume first is vertices, second is triangles
    if nodes is None:
        nodes = np.array(darrays[0].data)
    if tris is None:
        if len(darrays) > 1:
            tris = np.array(darrays[1].data)
        else:
            tris = np.array([])

    return nodes, tris


def _parse_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Wavefront OBJ (``.obj``) file.

    Extracts vertex positions (``v`` lines) and face indices (``f`` lines).
    Face indices are 1-based in OBJ files and are converted to 0-based.
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                # Vertex: v x y z
                if len(parts) >= 4:
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
            elif parts[0] == "f":
                # Face: f v1 v2 v3 ... (1-based, may include /vt/vn)
                indices: list[int] = []
                for p in parts[1:]:
                    idx_str = p.split("/")[0]
                    indices.append(int(idx_str) - 1)
                    if len(indices) == 3:
                        break
                if len(indices) == 3:
                    faces.append(indices)

    return np.array(vertices), np.array(faces)


def _parse_mz3(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Surf Ice MZ3 (``.mz3``) surface file.

    Uses nibabel for loading. Falls back to a binary parser for
    raw MZ3 files if nibabel cannot handle them.
    """
    try:
        import nibabel as nib

        img = nib.load(path)
        # MZ3 stores vertices and faces in aggregate data
        if hasattr(img, "agg_data"):
            data = img.agg_data()
            if len(data) >= 2:
                return np.array(data[0]), np.array(data[1])

        # Try accessing via dataobj
        if hasattr(img, "dataobj"):
            data = np.asarray(img.dataobj)
            if len(data) >= 2:
                return np.array(data[0]), np.array(data[1])

    except Exception:
        pass

    # Fallback: parse as binary MZ3
    import struct

    with open(path, "rb") as f:
        magic = f.read(3)
        if magic != b"MZ3":
            raise ValueError(f"Not a valid MZ3 file: {path}")

        # Read header (total 18 bytes: 2 byte version, 4x uint32)
        _version = struct.unpack("<H", f.read(2))[0]
        nface = struct.unpack("<I", f.read(4))[0]
        nvert = struct.unpack("<I", f.read(4))[0]
        _nskip = struct.unpack("<I", f.read(4))[0]

        # Read vertices (nvert * 3 floats)
        vert_data = struct.unpack(
            f"<{nvert * 3}f", f.read(nvert * 3 * 4)
        )
        vertices = np.array(vert_data, dtype=np.float64).reshape(nvert, 3)

        # Read faces (nface * 3 ints)
        face_data = struct.unpack(
            f"<{nface * 3}I", f.read(nface * 3 * 4)
        )
        faces = np.array(face_data, dtype=np.int32).reshape(nface, 3)

    return vertices, faces
