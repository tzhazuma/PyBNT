"""Predefined camera positions and view layouts with BNV parity."""

# Standard neuroimaging view presets (BNV-compatible).
# Each entry maps to a pyvista camera (position, focal_point) pair.
# The view-up vector defaults to (0, 0, 1) unless overridden.
CAMERA_PRESETS = {
    # --- Lateral views ---
    "lateral_left": {"position": (-400, 0, 0), "focal_point": (0, 0, 0)},
    "lateral_right": {"position": (400, 0, 0), "focal_point": (0, 0, 0)},

    # --- Medial views ---
    "medial_left": {"position": (0, -400, 0), "focal_point": (0, 0, 0)},
    "medial_right": {"position": (0, 400, 0), "focal_point": (0, 0, 0)},

    # --- Ventral / Dorsal ---
    "ventral": {"position": (0, 0, -400), "focal_point": (0, 0, 0)},
    "dorsal": {"position": (0, 0, 400), "focal_point": (0, 0, 0)},

    # --- Anterior / Posterior ---
    "anterior": {"position": (0, -400, 0), "focal_point": (0, 0, 0)},
    "posterior": {"position": (0, 400, 0), "focal_point": (0, 0, 0)},

    # --- Sagittal (BNV-standard naming) ---
    "sagittal_left": {
        "position": (-400, 0, 0),
        "focal_point": (0, 0, 0),
        "viewup": (0, 0, 1),
    },
    "sagittal_right": {
        "position": (400, 0, 0),
        "focal_point": (0, 0, 0),
        "viewup": (0, 0, 1),
    },

    # --- Axial / Transverse ---
    "axial_superior": {
        "position": (0, 0, 400),
        "focal_point": (0, 0, 0),
        "viewup": (0, -1, 0),
    },
    "axial_inferior": {
        "position": (0, 0, -400),
        "focal_point": (0, 0, 0),
        "viewup": (0, -1, 0),
    },

    # --- Coronal / Frontal ---
    "coronal_anterior": {
        "position": (0, -400, 0),
        "focal_point": (0, 0, 0),
        "viewup": (0, 0, 1),
    },
    "coronal_posterior": {
        "position": (0, 400, 0),
        "focal_point": (0, 0, 0),
        "viewup": (0, 0, 1),
    },
}


def apply_layout(plotter, layout_name):
    """Set camera to a predefined layout.

    Args:
        plotter: pyvista.Plotter instance.
        layout_name: Key from CAMERA_PRESETS.
    """
    layout = CAMERA_PRESETS.get(layout_name)
    if layout is None:
        raise ValueError(f"Unknown layout: {layout_name}")
    viewup = layout.get("viewup", (0, 0, 1))
    plotter.camera_position = [
        layout["position"],
        layout["focal_point"],
        viewup,
    ]


def medial_view(plotter):
    """Medial view — from inside looking out."""
    plotter.camera_position = [(0, 200, 0), (0, 0, 0), (0, 0, 1)]


def six_view(plotter, surfaces, nodes=None, edges=None):
    """Render 6 standard neuro views in a 2x3 subplot layout.

    Views: lateral_left | lateral_right | medial_left |
           ventral | dorsal | posterior

    Args:
        plotter: pyvista.Plotter instance (shape will be set to (2, 3)).
        surfaces: list of surface meshes (one per hemisphere is typical).
        nodes: optional list of node dicts with x, y, z, size keys.
        edges: optional edge connectivity matrix (NxN).
    """
    import pyvista as pv

    view_names = [
        "lateral_left", "lateral_right",
        "medial_left", "ventral",
        "dorsal", "posterior",
    ]

    plotter.subplot(2, 3)
    for i, view_name in enumerate(view_names):
        row = i // 3
        col = i % 3
        plotter.subplot(row, col)
        for surface in surfaces:
            plotter.add_mesh(surface, show_edges=True)
        if nodes:
            for node in nodes:
                ball = pv.Sphere(
                    radius=node.get("size", 1.0),
                    center=(node["x"], node["y"], node["z"]),
                )
                plotter.add_mesh(ball, color="red")
        if edges is not None:
            _draw_edges_in_plotter(plotter, nodes, edges)
        apply_layout(plotter, view_name)


def _draw_edges_in_plotter(plotter, nodes, edges):
    """Draw undirected edges between nodes within a subplot.

    Args:
        plotter: pyvista.Plotter instance.
        nodes: list of node dicts.
        edges: NxN numpy array of edge weights.
    """
    import pyvista as pv

    m, n = edges.shape
    for i in range(m):
        for j in range(n):
            if edges[i, j] > 0:
                node1 = nodes[i]
                node2 = nodes[j]
                line = pv.Line(
                    pointa=(node1["x"], node1["y"], node1["z"]),
                    pointb=(node2["x"], node2["y"], node2["z"]),
                )
                plotter.add_mesh(line, color="blue")


def medium_view(plotter, surface_path):
    """Render lateral and medial views of a single hemisphere side-by-side.

    Loads the surface from *surface_path*, creates two subplot panes
    showing lateral (left) and medial (right) perspectives.

    Args:
        plotter: pyvista.Plotter instance (shape set to (1, 2)).
        surface_path: path to .nv, .pial, .gii, or .obj surface file.
    """
    import numpy as np
    import pyvista as pv
    from pybnt.core.surface import parse_surface

    nodes, tris = parse_surface(surface_path)
    mesh = pv.make_tri_mesh(np.array(nodes), np.array(tris))

    plotter.subplot(1, 2)

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, show_edges=True)
    apply_layout(plotter, "lateral_left")

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, show_edges=True)
    medial_view(plotter)


def double_brain(plotter, surface1_path, surface2_path, label1="", label2=""):
    """Render two brain surfaces side-by-side for comparison.

    Loads both surfaces and places them in a (1, 2) subplot grid with
    optional text labels above each hemisphere.

    Args:
        plotter: pyvista.Plotter instance (shape set to (1, 2)).
        surface1_path: path to the first surface file.
        surface2_path: path to the second surface file.
        label1: text label for the first brain.
        label2: text label for the second brain.
    """
    import numpy as np
    import pyvista as pv
    from pybnt.core.surface import parse_surface

    nodes1, tris1 = parse_surface(surface1_path)
    mesh1 = pv.make_tri_mesh(np.array(nodes1), np.array(tris1))

    nodes2, tris2 = parse_surface(surface2_path)
    mesh2 = pv.make_tri_mesh(np.array(nodes2), np.array(tris2))

    plotter.subplot(1, 2)

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh1, show_edges=True)
    if label1:
        plotter.add_text(label1, position="upper_edge", font_size=12)
    apply_layout(plotter, "lateral_left")

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh2, show_edges=True)
    if label2:
        plotter.add_text(label2, position="upper_edge", font_size=12)
    apply_layout(plotter, "lateral_left")
