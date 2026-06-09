"""3D brain surface/node/edge visualization using pyvista."""


def draw_surface(surface_path, image_path=None, voxel=False, smooth=False,
                 light="default", show=True, colormap="white", opacity=1.0):
    """Draw brain surface mesh.

    Args:
        surface_path: path to .nv or .pial surface file.
        image_path: optional path to volumetric image for overlay.
        voxel: if True, overlay the image as voxels.
        smooth: apply Laplacian smoothing to the mesh.
        light: name of lighting configuration.
        show: if True, immediately display the plotter window.
        colormap: name of colormap or list of RGB tuples.
        opacity: surface opacity (0.0 to 1.0).

    Returns:
        pyvista.Plotter instance.
    """
    import numpy as np
    import pyvista as pv
    from pybnt.core.surface import parse_surface

    nodes, tris = parse_surface(surface_path)
    img = pv.make_tri_mesh(np.array(nodes), np.array(tris))

    ax = pv.Plotter()
    ax.add_mesh(img, color="white", show_edges=True, opacity=opacity)
    ax.enable_anti_aliasing("msaa", 16)

    if smooth:
        img = img.smooth()

    from pybnt.visualization.lighting import apply_lighting
    apply_lighting(ax, light)

    if image_path is not None and voxel:
        from pybnt.core.io import readimage
        ax.add_mesh(draw_voxels(readimage(image_path), show=False))

    if show:
        ax.show()
    return ax


def draw_nodes(node_path, edge_path=None, surface_path=None, image_path=None,
               voxel=False, node_type="all", color_tuple=("red", "blue", "white"),
               smooth=False, proj_type="nearest", show=True, node_scale=1.0):
    """Draw nodes with optional edges and surface.

    Args:
        node_path: path to node definition file.
        edge_path: optional path to edge connectivity file.
        surface_path: optional path to surface mesh for projection.
        image_path: optional path to volumetric image for overlay.
        voxel: if True, overlay the image as voxels.
        node_type: "all", "node", or "nande" to control what is rendered.
        color_tuple: (node_color, edge_color, surface_color).
        smooth: apply Laplacian smoothing to the surface.
        proj_type: projection type for node-to-surface mapping.
        show: if True, immediately display the plotter window.
        node_scale: scaling factor for node sphere radius.

    Returns:
        pyvista.Plotter instance.
    """
    import numpy as np
    import pyvista as pv
    from pybnt.core.nodes import parse_node_text
    from pybnt.core.edges import parse_edge_text

    nodes = parse_node_text(node_path)
    edges = None
    if edge_path is not None:
        edges = parse_edge_text(edge_path)

    ax = pv.Plotter()
    for node in nodes:
        ball = pv.Sphere(
            radius=node["size"] * node_scale,
            center=(node["x"], node["y"], node["z"]),
        )
        ax.add_mesh(ball, color=color_tuple[0])

    if node_type != "node" and edges is not None:
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
                    ax.add_mesh(line, color=color_tuple[1])

    if surface_path is not None:
        from pybnt.core.surface import parse_surface
        from pybnt.core.projection import Triangle, Surfaces

        n, t = parse_surface(surface_path)
        img = pv.make_tri_mesh(np.array(n), np.array(t))
        img = img.extract_surface(nonlinear_subdivision=20)

        if smooth:
            img = img.smooth()

        if node_type == "all":
            tl = []
            for triindex in t:
                tri = Triangle(
                    (n[triindex[0]], n[triindex[1]], n[triindex[2]]),
                    nodesindex=triindex,
                )
                tl.append(tri)
            sf = Surfaces(tl)
            sf.proj_surface_all(ps=nodes, type=proj_type)
            val = np.zeros(img.points.shape[0])
            for tri in sf:
                for i in range(3):
                    val[tri.nodesindex[i]] += tri.nodes[i].value
            img.point_data["value"] = val
            img.integrate_data()

        if node_type not in ("nande", "node"):
            ax.add_mesh(
                img, color=color_tuple[2], scalars="value", show_edges=True
            )

    if image_path is not None and voxel:
        from pybnt.core.io import readimage
        ax.add_mesh(draw_voxels(readimage(image_path), show=False))

    if show:
        ax.show()
    return ax


def draw_voxels(voxel_array, show=False):
    """Draw a 3D voxel grid.

    Args:
        voxel_array: 3D numpy array of voxel data.
        show: if True, immediately display the plot.

    Returns:
        pyvista.UniformGrid instance.
    """
    import pyvista as pv

    img = pv.UniformGrid()
    img.dimensions = voxel_array.shape
    img.origin = (0, 0, 0)
    img.spacing = (1, 1, 1)
    img.point_arrays["voxel"] = voxel_array.flatten(order="F")

    if show:
        img.plot(show_edges=True)
    return img


def draw_roi(image, roi_mask, show=False):
    """Draw ROI from image with mask.

    Voxels outside the mask are set to zero before rendering.

    Args:
        image: 3D numpy array of image data.
        roi_mask: boolean / integer mask array matching image shape.
        show: if True, immediately display the plot.

    Returns:
        pyvista.UniformGrid instance.
    """
    image[roi_mask == 0] = 0
    return draw_voxels(image, show)


def draw_roi_with_surface(surface_path, image, roi_mask, show=True):
    """Draw ROI overlay on top of a brain surface.

    Args:
        surface_path: path to surface mesh file.
        image: 3D numpy array of image data.
        roi_mask: mask array for the ROI.
        show: if True, immediately display the plotter window.

    Returns:
        pyvista.Plotter instance.
    """
    ax = draw_surface(surface_path, show=False)
    roi = draw_roi(image, roi_mask, show=False)
    ax.add_mesh(roi, opacity=0.5)

    if show:
        ax.show()
    return ax


def add_labels(plotter, node_list, font_size=12, color="black", offset=(0, 0, 0)):
    """Add text labels at node positions in the 3D scene.

    Each node in *node_list* must contain "x", "y", "z" keys for
    position and a "name" key for the displayed text.

    Args:
        plotter: pyvista.Plotter instance.
        node_list: list of dicts with x, y, z, name keys.
        font_size: text size (default 12).
        color: text color (default "black").
        offset: (dx, dy, dz) world-space offset from node position.
    """
    for node in node_list:
        x = node["x"] + offset[0]
        y = node["y"] + offset[1]
        z = node["z"] + offset[2]
        plotter.add_point_labels(
            [(x, y, z)],
            [node.get("name", "")],
            font_size=font_size,
            text_color=color,
            point_size=0,
        )


def draw_directed_edges(plotter, nodes, edges, color="blue"):
    """Draw directed edges as cylinders with arrow-head cones.

    For each non-zero entry in the NxN *edges* matrix, a directed edge
    is drawn from nodes[i] to nodes[j].  The shaft is a cylinder and
    the tip is a cone pointing toward the target.

    Args:
        plotter: pyvista.Plotter instance.
        nodes: list of node dicts with x, y, z keys.
        edges: NxN numpy array where edges[i,j] > 0 means a connection.
        color: line/arrow color (default "blue").
    """
    import numpy as np
    import pyvista as pv

    m, n = edges.shape
    for i in range(m):
        for j in range(n):
            if edges[i, j] <= 0:
                continue
            p1 = np.array([nodes[i]["x"], nodes[i]["y"], nodes[i]["z"]])
            p2 = np.array([nodes[j]["x"], nodes[j]["y"], nodes[j]["z"]])
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction = direction / length

            # Cylinder shaft (80% of edge length)
            shaft_length = length * 0.8
            shaft_start = p1
            shaft_center = shaft_start + direction * shaft_length / 2
            shaft = pv.Cylinder(
                center=shaft_center,
                direction=direction,
                radius=0.5,
                height=shaft_length,
            )
            plotter.add_mesh(shaft, color=color)

            # Cone arrow head at target end
            cone_pos = p1 + direction * shaft_length
            cone = pv.Cone(
                center=cone_pos,
                direction=direction,
                radius=1.0,
                height=length * 0.2,
            )
            plotter.add_mesh(cone, color=color)


def set_material(plotter, mesh, shiny=True, metal=False):
    """Configure the material properties of a mesh.

    Provides BNV-style surface appearance controls.

    Args:
        plotter: pyvista.Plotter instance.
        mesh: pyvista mesh to modify (added via add_mesh).
        shiny: Enable specular highlights (default True).
        metal: Enable metallic appearance (default False).
    """
    if shiny:
        plotter.add_mesh(mesh, specular=0.5, specular_power=20, smooth_shading=True)
    if metal:
        plotter.add_mesh(mesh, metallic=1.0, roughness=0.3)
