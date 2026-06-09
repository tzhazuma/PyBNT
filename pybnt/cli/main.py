"""PyBrainViewer Command-Line Interface."""
import click
from pathlib import Path


@click.group()
@click.version_option(version="0.2.0")
def cli():
    """PyBrainViewer - AI-Powered Multi-Function Brain Visual Tool."""
    pass


@cli.group()
def visualize():
    """Visualize brain surfaces, nodes, and edges."""
    pass


@visualize.command()
@click.argument("surface_path", type=click.Path(exists=True))
@click.option("--smooth", is_flag=True, help="Smooth the surface mesh")
@click.option("--light", default="default", help="Lighting preset")
@click.option("--colormap", default="white", help="Surface colormap")
@click.option("--opacity", default=1.0, type=float, help="Surface opacity")
@click.option("--output", "-o", type=click.Path(), help="Save screenshot to file")
def surface(surface_path, smooth, light, colormap, opacity, output):
    """Render a 3D brain surface from a .nv or .pial file."""
    from pybnt.visualization.plotter import draw_surface
    ax = draw_surface(surface_path, smooth=smooth, light=light,
                       colormap=colormap, opacity=opacity, show=(output is None))
    if output:
        ax.screenshot(output)
        click.echo(f"Screenshot saved to {output}")


@visualize.command()
@click.argument("node_path", type=click.Path(exists=True))
@click.option("--edge-path", "-e", type=click.Path(exists=True), help="Edge file")
@click.option("--surface-path", "-s", type=click.Path(exists=True), help="Surface file")
@click.option("--node-color", default="red", help="Node color")
@click.option("--edge-color", default="blue", help="Edge color")
@click.option("--surface-color", default="white", help="Surface color")
@click.option("--node-scale", default=1.0, type=float, help="Node scaling")
@click.option("--output", "-o", type=click.Path(), help="Save screenshot")
def nodes(node_path, edge_path, surface_path, node_color, edge_color,
           surface_color, node_scale, output):
    """Render brain nodes with optional edges and surface."""
    from pybnt.visualization.plotter import draw_nodes
    colors = (node_color, edge_color, surface_color)
    ax = draw_nodes(node_path, edge_path, surface_path,
                     color_tuple=colors, node_scale=node_scale,
                     show=(output is None))
    if output:
        ax.screenshot(output)


@cli.group()
def process():
    """Process brain imaging data."""
    pass


@process.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True)
def convert(input_path, output):
    """Convert between DICOM, NIfTI, and image formats."""
    from pybnt.core.io import dicom_to_nifti
    ext = Path(input_path).suffix.lower()
    out_ext = Path(output).suffix.lower()

    if ext in ('.dcm', '.dicom') and out_ext in ('.nii', '.nii.gz'):
        dicom_to_nifti(input_path, output)
        click.echo(f"Converted {input_path} → {output}")
    else:
        click.echo(f"Unsupported conversion: {ext} → {out_ext}", err=True)


@process.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output path")
@click.option("--method", default="correlation", help="Connectivity method")
def compute_connectivity(image_path, output, method):
    """Compute brain connectivity matrix from fMRI data."""
    from pybnt.processing.connectivity import compute_correlation_matrix
    matrix = compute_correlation_matrix(image_path)
    if output:
        import numpy as np
        np.savetxt(output, matrix)
        click.echo(f"Connectivity matrix saved to {output}")
    else:
        click.echo(f"Matrix shape: {matrix.shape}")


@process.command()
@click.argument("fixed_image", type=click.Path(exists=True))
@click.argument("moving_image", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="./registration_output")
def register(fixed_image, moving_image, output_dir):
    """Register moving image to fixed image."""
    from pybnt.external.elastix import register_elastix
    result = register_elastix(fixed_image, moving_image, output_dir)
    click.echo(f"Registration complete: {result}")


@process.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), default="./superres_output")
@click.option("--scale", default=2, type=int, help="Super-resolution scale factor")
def superres(input_dir, output_dir, scale):
    """Apply AI super-resolution to brain images."""
    from pybnt.processing.superres import run_superres
    run_superres(input_dir, output_dir, output_dir)
    click.echo(f"Super-resolution complete. Output: {output_dir}")


@cli.group()
def ai():
    """AI-powered tools."""
    pass


@ai.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./segmentation_output")
def segment(image_path, output):
    """Segment brain regions using AI."""
    from pybnt.ai.segmentation import segment_brain
    result = segment_brain(image_path, output)
    click.echo(f"Segmentation complete: {result}")


@ai.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./corrected_output")
def correct(image_path, output):
    """Correct/denoise brain images."""
    from pybnt.ai.correction import correct_image
    import numpy as np
    from pybnt.core.utils import readimage
    img = readimage(image_path)
    result = correct_image(img, method="denoise")
    from pybnt.core.io import image_to_nifti
    image_to_nifti(result, output)
    click.echo(f"Correction complete: {output}")


@ai.command()
@click.option("--message", "-m", required=True, help="Question to ask")
@click.option("--api-key", "-k", help="API key (or set DASHSCOPE_API_KEY)")
@click.option("--model", default="qwen-plus", help="LLM model name")
def ask(message, api_key, model):
    """Ask an AI assistant about brain science."""
    from pybnt.ai.llm import ask_llm
    msgs = [{"role": "user", "content": message}]
    response = ask_llm(msgs, api_key=api_key or "", model=model)
    if response:
        click.echo(response)
    else:
        click.echo("Failed to get response. Check API key.", err=True)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--layout", default="lateral_left", help="View layout")
@click.option("--colormap", default="jet", help="Colormap name")
@click.option("--output", "-o", type=click.Path(), default="surface_volume.png")
def volume(input_path, layout, colormap, output):
    """Map volume data to brain surface and render."""
    from pybnt.core.volume import map_volume_to_surface
    from pybnt.core.surface import parse_surface
    import numpy as np

    from importlib import resources
    import pybnt.data.templates as _templates
    surface_path = str(resources.files(_templates) / "BrainMesh_ICBM152.nv")
    vertex_data = map_volume_to_surface(input_path, surface_path)
    nodes, tris = parse_surface(surface_path)

    import pyvista as pv
    mesh = pv.make_tri_mesh(np.array(nodes), np.array(tris))
    mesh.point_data["value"] = vertex_data

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="value", cmap=colormap, show_edges=True)

    from pybnt.visualization.layouts import apply_layout
    apply_layout(plotter, layout)

    plotter.screenshot(output)
    click.echo(f"Volume-to-surface map saved: {output}")


if __name__ == "__main__":
    cli()
