"""
Surface-based projection of node values onto triangle meshes.

Classes
-------
Triangle
    A single triangle on a brain surface, with region and value support.
Surfaces
    A collection of ``Triangle`` objects with projection methods.

Projection types supported (``proj_all`` / ``proj_surface_all``)
---------------------------------------------------------------
- ``"all"``        - Project onto every triangle where the node lands.
- ``"nearest"``    - Project onto the single nearest triangle.
- ``"surround"``   - Project onto all triangles within ``neard`` distance.
- ``"minium"``     - Nearest + neighbours within range, take the minimum.
- ``"maxium"``     - Nearest + neighbours within range, take the maximum.
- ``"average"``    - Nearest + neighbours within range, take the mean.
- ``"gaussian_nearest"``  - Like nearest but uses Gaussian-weighted distance.
- ``"gaussian_minium"``   - Like minium but uses Gaussian-weighted distance.
- ``"gaussian_maxium"``   - Like maxium but uses Gaussian-weighted distance.
- ``"gaussian_average"``  - Like average but uses Gaussian-weighted distance.
- ``"interpolate"``       - Bilinear-style interpolation within the triangle.
"""

from math import acos
from typing import Optional, Union

import numpy as np

from pybnt.core.logconf import logger
from pybnt.core.utils import dis, gaussian_dis

# Type alias for 3D position
Pos = tuple[float, float, float]


class Triangle:
    """A single triangle on a brain surface mesh."""

    def __init__(
        self,
        nodes: tuple[Pos, Pos, Pos],
        brain_re: Optional[str] = None,
        nodesindex: Optional[tuple[int, int, int]] = None,
    ) -> None:
        self.nodes = np.array(list(nodes))
        self.region: Optional[str] = brain_re
        self.value: float = 0.0
        self.nodesindex = nodesindex

    def calculate_mid(self) -> np.ndarray:
        """Return the midpoint (centroid) of the triangle."""
        return np.mean(self.nodes, axis=0)

    def clear(self) -> None:
        """Reset the triangle value to 0."""
        self.value = 0.0

    def clear_region(self) -> None:
        """Remove the assigned brain region."""
        self.region = ""

    def proj_value(self, value: float) -> None:
        """Accumulate a projected value."""
        self.value += value

    def set_region(self, region: str) -> None:
        """Set the brain region label."""
        self.region = region

    def calculate_h_line(self) -> tuple[float, float, float]:
        """Compute the direction vector of the triangle's horizontal line.

        This is used internally for projection calculations.
        """
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        n3 = self.nodes[2]
        l1 = n2 - n1
        l2 = n3 - n1
        x1, y1, z1 = l1[0:3]
        x2, y2, z2 = l2[0:3]
        k = z1 * y2 - z2 * y1
        if k != 0:
            x3 = 1.0
            z3 = (-x1 * y2 + x2 * y1) / k
            if y1 != 0:
                y3 = (-x1 - z1 * z3) / y1
                return (x3, y3, z3)
            else:
                y3 = (-x2 - z2 * z3) / y2
                return (x3, y3, z3)
        else:
            if x1 * y2 != x2 * y1:
                x3 = 0.0
                if not (z1 == 0 and y1 == 0):
                    return (x3, z1, -y1)
                else:
                    return (x3, z2, -y2)
            else:
                if not (x1 == 0 and y1 == 0):
                    return (y1, -x1, 0.0)
                else:
                    return (y2, -x2, 0.0)

    def proj_pos(self, p: Union[Pos, np.ndarray]) -> np.ndarray:
        """Project a point onto the triangle's plane."""
        proj_line = np.array(self.calculate_h_line())
        n1 = self.nodes[0]
        l = np.array(p) - n1
        s = -l.dot(proj_line)
        q = proj_line.dot(proj_line)
        t = s / q
        return np.array(p) + t * proj_line

    @staticmethod
    def veclen(v: np.ndarray) -> float:
        """Compute the Euclidean length of a vector."""
        return float(np.sqrt(v.dot(v)))

    def vecang(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute the angle (in radians) between two vectors."""
        denom = self.veclen(v1) * self.veclen(v2)
        s = max(-1.0, min(1.0, v1.dot(v2) / denom))
        return float(acos(s))

    def pos_in(self, p: Pos) -> bool:
        """Check whether a point lies inside the triangle (2D barycentric test)."""
        n1, n2, n3 = self.nodes[0:3]
        p_arr = np.array(p)
        l1 = n2 - n1
        l2 = n3 - n1
        d = p_arr - n1
        s = self.vecang(l1, l2)
        s1 = self.vecang(l1, d)
        s2 = self.vecang(l2, d)
        if abs(s - (s1 + s2)) > 1e-10:
            return False

        l1 = n3 - n2
        l2 = n1 - n2
        d = p_arr - n2
        s = self.vecang(l1, l2)
        s1 = self.vecang(l1, d)
        s2 = self.vecang(l2, d)
        if abs(s - (s1 + s2)) > 1e-10:
            return False

        return True

    def need_proj(self, p: dict, regionbound: bool = True) -> bool:
        """Determine whether a node should be projected onto this triangle.

        A node is projected if its region label matches the triangle's
        region (or region bounding is disabled) and its projected position
        falls inside the triangle.
        """
        if p["label"] != self.region and regionbound:
            return False
        proj_p = self.proj_pos((p["x"], p["y"], p["z"]))
        return self.pos_in(proj_p)

    @staticmethod
    def dis(p1: Union[Pos, np.ndarray], p2: Union[Pos, np.ndarray]) -> float:
        """Compute the Euclidean distance between two points."""
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        dif = p2_arr - p1_arr
        return float(np.sqrt(dif.dot(dif)))

    def pos_dis(self, p: dict) -> float:
        """Compute the distance from a node to its projection on this triangle."""
        proj_p = np.array(self.proj_pos((p["x"], p["y"], p["z"])))
        ori_p = np.array((p["x"], p["y"], p["z"]))
        return self.dis(proj_p, ori_p)

    def mean_node(self) -> np.ndarray:
        """Return the mean position of the triangle's three vertices."""
        return np.mean(np.array(self.nodes), axis=0)


class Surfaces:
    """A collection of ``Triangle`` objects representing a brain surface."""

    def __init__(
        self,
        triangles: list[Triangle],
        regionnodes: Optional[dict] = None,
    ) -> None:
        self.tri = triangles
        if regionnodes is not None:
            for triangle in self.tri:
                triangle.set_region(regionnodes[triangle.mean_node()])

    def proj_single(
        self,
        p: dict,
        proj_type: str = "all",
        neard: float = 10.0,
    ) -> None:
        """Project a single node's value onto the surface.

        Args:
            p: Node dict with ``x, y, z, value, label`` keys.
            proj_type: One of ``"all"``, ``"nearest"``, ``"surround"``.
            neard: Distance threshold for ``"surround"`` type.
        """
        if proj_type == "all":
            for triangle in self.tri:
                if triangle.need_proj(p):
                    triangle.proj_value(p["value"])
        elif proj_type == "nearest":
            mindis = 9999999.0
            minidx = 0
            for index, triangle in enumerate(self.tri):
                if triangle.need_proj(p):
                    d = triangle.pos_dis(p)
                    if d < mindis:
                        mindis = d
                        minidx = index
            self.tri[minidx].proj_value(p["value"])
        elif proj_type == "surround":
            for triangle in self.tri:
                d = triangle.pos_dis(p)
                if d <= neard and triangle.need_proj(p):
                    triangle.proj_value(p["value"])
        else:
            raise ValueError("Projection type error!")

    def proj_all(
        self,
        ps: list[dict],
        proj_type: str = "all",
    ) -> None:
        """Project multiple nodes onto the surface.

        Args:
            ps: List of node dicts.
            proj_type: Passed through to ``proj_single``.
        """
        for p in ps:
            self.proj_single(p, proj_type)

    def proj_surface_all(
        self,
        ps: list[dict],
        proj_type: str = "nearest",
        disrange: float = 10.0,
        regionbound: bool = False,
    ) -> None:
        """Project node values onto every triangle on the surface.

        For each triangle the closest (or otherwise selected) node value
        is determined and assigned.

        Args:
            ps: List of node dicts.
            proj_type: Projection strategy. Supported values:

                * ``"nearest"`` - single nearest node.
                * ``"minium"`` - nearest + neighbours, minimum value.
                * ``"maxium"`` - nearest + neighbours, maximum value.
                * ``"average"`` - nearest + neighbours, mean value.
                * ``"gaussian_nearest"`` - Gaussian-weighted nearest.
                * ``"gaussian_minium"`` - Gaussian-weighted minimum.
                * ``"gaussian_maxium"`` - Gaussian-weighted maximum.
                * ``"gaussian_average"`` - Gaussian-weighted average.
                * ``"interpolate"`` - bilinear-style interpolation.
            disrange: Extra distance tolerance for neighbour search.
            regionbound: Whether to restrict by region label.
        """
        for triangle in self.tri:
            triangle.clear()
            mean_node = triangle.mean_node()

            if proj_type == "nearest":
                mindis = 9999999.0
                minidx = 0
                for index, p in enumerate(ps):
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                        minidx = index
                triangle.proj_value(ps[minidx]["value"])

            elif proj_type == "minium":
                mindis = 9999999.0
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors: list[float] = []
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(min(neighbors))

            elif proj_type == "maxium":
                mindis = 9999999.0
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors = []
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(max(neighbors))

            elif proj_type == "average":
                mindis = 9999999.0
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors = []
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(float(np.mean(neighbors)))

            elif proj_type == "gaussian_nearest":
                mindis = 9999999.0
                minidx = 0
                for index, p in enumerate(ps):
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                        minidx = index
                triangle.proj_value(ps[minidx]["value"])

            elif proj_type == "gaussian_minium":
                mindis = 9999999.0
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors = []
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(min(neighbors))

            elif proj_type == "gaussian_maxium":
                mindis = 9999999.0
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors = []
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(max(neighbors))

            elif proj_type == "gaussian_average":
                mindis = 9999999.0
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors = []
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(p["value"])
                if neighbors:
                    triangle.proj_value(float(np.mean(neighbors)))

            elif proj_type == "interpolate":
                mindis = 9999999.0
                for p in ps:
                    d = gaussian_dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d < mindis and triangle.need_proj(p, regionbound):
                        mindis = d
                disr = mindis + disrange
                neighbors: list[tuple[float, np.ndarray]] = []
                for p in ps:
                    d = dis(mean_node, (p["x"], p["y"], p["z"]))
                    if d <= disr and triangle.need_proj(p, regionbound):
                        neighbors.append(
                            (p["value"], np.array([p["x"], p["y"], p["z"]]))
                        )
                out = np.zeros(3)
                for value, pos in neighbors:
                    proj_pos = triangle.proj_pos(pos)
                    l = proj_pos - mean_node
                    out += value * l
                triangle.proj_value(float(np.sum(out)))

            else:
                logger.warning(f"Unknown projection type: {proj_type!r}")


# Backward-compatible aliases for old method names
Triangle.caculate_mid = Triangle.calculate_mid  # type: ignore[attr-defined]
Triangle.cauclate_h_line = Triangle.calculate_h_line  # type: ignore[attr-defined]
