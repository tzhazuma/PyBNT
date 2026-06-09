"""
Node file parsing and node-level computations.

Functions for reading node text files, computing node positions,
values, sizes, ROI statistics, and connectivity-based values.
"""

from copy import deepcopy
from typing import Optional, Union

import numpy as np

from pybnt.core.logconf import logger

# Type aliases for clarity
NodeList = list[dict[str, Union[float, str]]]
NodeTuple = tuple[float, float, float, float]


def calculate_node_pos(
    brainareanodes: list[Union[tuple[float, float, float], tuple[float, float, float, float]]],
) -> np.ndarray:
    """Calculate the mean position of nodes in a brain area.

    Args:
        brainareanodes: List of tuples, each containing (x, y, z) or (x, y, z, value).

    Returns:
        Mean (x, y, z) position as a numpy array.
    """
    arr = np.array(brainareanodes)
    return np.mean(arr[:, 0:3], axis=0)


def calculate_node_value(
    brainareanodes: list[tuple[float, float, float, float]],
    valueaxis: int = 3,
) -> float:
    """Calculate the mean value of nodes in a brain area.

    Args:
        brainareanodes: List of tuples, each containing (x, y, z, value).
        valueaxis: Column index for the value (default 3).

    Returns:
        Mean value as a float.
    """
    values = np.array(brainareanodes)[:, valueaxis]
    return float(np.mean(values))


def calculate_node_size(
    brainareanodes: list[tuple[float, float, float, float]],
    fac: float = 0.01,
) -> float:
    """Calculate the size of nodes in a brain area.

    Args:
        brainareanodes: List of tuples, each containing (x, y, z, value).
        fac: Scaling factor (default 0.01).

    Returns:
        Size as ``len(nodes) * fac``.
    """
    return len(brainareanodes) * fac


def parse_node_file(
    path: str,
    mode: str = "continue",
) -> Optional[list[dict[str, Union[float, str]]]]:
    """Parse a node text file.

    File format (tab-separated)::

        x   y   z   value   size   label

    Args:
        path: Path to the node file.
        mode: Error handling mode. ``"continue"`` skips malformed lines;
              any other value returns ``None`` on the first error.

    Returns:
        List of node dicts with keys ``x``, ``y``, ``z``, ``value``,
        ``size``, ``label``, or ``None`` on error.
    """
    outnodes: list[dict[str, Union[float, str]]] = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split("\t")
        try:
            x, y, z, value, size, label = data
            node: dict[str, Union[float, str]] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "value": float(value),
                "size": float(size),
                "label": label,
            }
            outnodes.append(node)
        except Exception:
            logger.warning(f"Data error at {data}")
            if mode == "continue":
                continue
            else:
                return None

    return outnodes


def roi_compute(
    roinodes: list[tuple[float, float, float, float]],
    scalfac: float = 1.0,
    mode: str = "average",
) -> Optional[float]:
    """Compute a summary statistic over ROI nodes.

    Args:
        roinodes: List of (x, y, z, value) tuples.
        scalfac: Scale factor applied to the result.
        mode: ``"average"``, ``"max"``, or ``"min"``.

    Returns:
        Scaled summary value, or ``None`` if mode is unsupported.
    """
    values = np.array(roinodes)[:, 3]
    if mode == "average":
        return float(np.mean(values) * scalfac)
    elif mode == "max":
        return float(np.max(values) * scalfac)
    elif mode == "min":
        return float(np.min(values) * scalfac)
    else:
        logger.warning("mode is not supported!")
        return None


def node2connect(
    nodelist: list[dict[str, Union[float, str]]],
    connectmatrix: np.ndarray,
) -> list[dict[str, Union[float, str]]]:
    """Compute node values from a connectivity matrix.

    Each node's value is set to its relative connectivity strength
    (row sum divided by total matrix sum).

    Args:
        nodelist: List of node dicts with ``x, y, z, value, size, label``.
        connectmatrix: Square connectivity matrix (nodes x nodes).

    Returns:
        New list of node dicts with updated ``value`` fields.
    """
    lin = np.sum(connectmatrix, axis=1) / np.sum(connectmatrix)
    outnodelist: list[dict[str, Union[float, str]]] = []
    for i, node in enumerate(nodelist):
        outnode = deepcopy(node)
        outnode["value"] = float(lin[i])
        outnodelist.append(outnode)
    return outnodelist


def calculate_node_value_from_image(
    image: np.ndarray,
    regionmask: np.ndarray,
    stat_type: str = "average",
) -> Optional[float]:
    """Calculate a summary value for a node from image data.

    Args:
        image: Image data array.
        regionmask: Boolean mask selecting the node's voxels.
        stat_type: ``"average"``, ``"max"``, ``"min"``, or
                   ``"filteraverage"`` (mean within ±1 std deviation).

    Returns:
        Summary value or ``None`` on error.
    """
    nodevoxels = image[regionmask]
    if stat_type == "average":
        return float(np.mean(nodevoxels))
    elif stat_type == "max":
        return float(np.max(nodevoxels))
    elif stat_type == "min":
        return float(np.min(nodevoxels))
    elif stat_type == "filteraverage":
        m = np.mean(nodevoxels)
        t = np.std(nodevoxels)
        filtered = nodevoxels[(nodevoxels > m - t) & (nodevoxels < m + t)]
        return float(np.mean(filtered))
    else:
        logger.warning(f"Unknown stat_type: {stat_type!r}")
        return None


# Backward-compatible aliases
nodevaluecauclate = calculate_node_value_from_image
cauclate_node_pos = calculate_node_pos
cauclate_node_value = calculate_node_value
cauclate_node_size = calculate_node_size
parse_node_text = parse_node_file
