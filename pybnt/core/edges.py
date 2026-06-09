"""
Edge file parsing and edge construction from node pairs.

Functions for reading edge/matrix text files and building
connectivity matrices from explicit node-pair lists.
"""

from typing import Optional

import numpy as np

from pybnt.core.logconf import logger
from pybnt.core.utils import isfloat


def parse_edge_file(
    path: str,
    mode: str = "continue",
) -> Optional[np.ndarray]:
    """Parse an edge/matrix text file.

    Each line contains space- or tab-separated numeric values.

    Args:
        path: Path to the edge file.
        mode: ``"continue"`` skips malformed lines; any other value
              returns ``None`` on the first error.

    Returns:
        Edge matrix as a 2D numpy array, or ``None`` on error.
    """
    outedges: list[list[float]] = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split("\t")
        try:
            numeric = [float(x) for x in data]
            outedges.append(numeric)
        except Exception:
            logger.warning(f"Data error at {data}")
            if mode == "continue":
                continue
            else:
                return None

    return np.array(outedges)


def make_edge_from_nodes(
    size: tuple[int, int],
    outpath: str,
    pairnodesl: list[tuple[int, int] | tuple[int, int, float]],
) -> Optional[np.ndarray]:
    """Build an edge matrix from a list of node-pair connections.

    For 2-element pairs ``(i, j)``, ``matrix[i, j]`` is set to 1.
    For 3-element pairs ``(i, j, w)``, ``matrix[i, j]`` is set to ``w``.
    The resulting matrix is written to ``outpath`` in space-separated
    row format.

    Args:
        size: Shape ``(rows, cols)`` of the output matrix.
        outpath: Path to write the matrix text file.
        pairnodesl: List of ``(i, j)`` or ``(i, j, w)`` node-pair entries.

    Returns:
        The edge matrix as a list of lists, or ``None`` on error.
    """
    try:
        arr = np.zeros(size)
    except Exception:
        logger.warning(f"shape error at size {size}")
        return None

    for pairnodes in pairnodesl:
        if len(pairnodes) == 2:
            try:
                arr[pairnodes[0], pairnodes[1]] = 1
            except Exception:
                logger.warning(f"nodes position error {pairnodes}")
                return None
        else:
            try:
                arr[pairnodes[0], pairnodes[1]] = pairnodes[2]
                if not isfloat(pairnodes[2]):
                    logger.warning(f"nodes value error at value {pairnodes[2]}")
                    return None
            except Exception:
                logger.warning(f"nodes position error {pairnodes}")
                return None

    result = arr.tolist()
    with open(outpath, "w") as f:
        for vec in result:
            f.write(" ".join([str(i) for i in vec]) + "\n")

    return np.array(result)


# Backward-compatible aliases
parse_edge_text = parse_edge_file
