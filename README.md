# PyBrainViewer (PyBNT)

**AI-Powered Multi-Function Brain Visual Tool** — A modern Python replacement for [BrainNet Viewer](https://www.nitrc.org/projects/bnv/).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-orange)](https://github.com/tzhazuma/PyBNT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()

## Overview

PyBrainViewer is a comprehensive, cross-platform tool for brain imaging data analysis and 3D visualization built from the ground up in Python. It parses and renders brain surfaces, network nodes, and connectivity edges using PyVista, and provides both classical and AI-powered processing pipelines for neuroimaging workflows.

The project was created as a modern, open-source replacement for the MATLAB-based BrainNet Viewer, with the same file format compatibility plus support for additional neuroimaging formats (GIFTI, FreeSurfer, OBJ, NIfTI, DICOM). Three interface modes let you pick the right tool for the job: a Click-based CLI for scripting, a Textual-based TUI for terminal browsing, and a full PyQt6 GUI for interactive visualization.

What you can do with PyBrainViewer:

- **3D Brain Surface Rendering** — Parse and display brain surfaces from .nv, .pial, .gii, and .obj files. Control color, opacity, smooth shading, edge display, and lighting.
- **Node and Edge Visualization** — Display brain network nodes as scaled spheres and edges as connectivity lines, with full control over size, color, and thickness. Supports directed networks with arrow-tipped edges.
- **Volume-to-Surface Mapping** — 9 projection algorithms including nearest neighbor, min/max, average, Gaussian-weighted variants, and bilinear interpolation, with region-bound filtering.
- **Image Processing** — DICOM to NIfTI conversion, min-max and z-score normalization, SNR/CNR/PSNR/SSIM/NRMSE quality metrics, and geometric transforms.
- **Connectivity Analysis** — fMRI correlation matrix computation via nilearn, supporting 4D NIfTI, directory inputs, raw arrays, and nibabel objects with automatic input type detection.
- **Image Registration** — Elastix-based linear and non-linear registration with automatic download support and motion correction for fMRI time series.
- **AI Super-Resolution** — EDSR-based convolutional neural network for brain image super-resolution, with training and inference pipelines. Includes a dual-input Combi variant for multi-modal data.
- **AI Segmentation and Correction** — Brain tissue segmentation with trained model inference and Otsu fallback. Image correction with denoising (non-local means) and motion artifact reduction.
- **LLM Assistant** — Brain science AI assistant via the Dashscope API (Qwen models), with a configurable API key and model selection.
- **FreeSurfer and SPM Integration** — Wrappers for FreeSurfer recon-all pipeline with automatic download, and SPM12 batch processing with pure-Python GLM fallback.
- **Three Interfaces** — CLI via Click, TUI via Textual, and GUI via PyQt6 with background threading for long processing tasks.

## Installation

### From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/tzhazuma/PyBNT.git
cd PyBNT
pip install -e .
```

### Using pip

```bash
pip install pybnt
```

### Dependencies

Core dependencies (auto-installed with pip):

`numpy`, `pyvista`, `PyQt6`, `opencv-python`, `nibabel`, `nilearn`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`, `pydicom`, `dicom2nifti`, `tqdm`, `pillow`, `numba`, `networkx`, `click`, `rich`, `textual`, `requests`, `dashscope`, `pymeshlab`

Optional AI dependencies (torch, tensorflow, transformers):

```bash
pip install pybnt[ai]
```

Optional external tool dependencies (pyelastix, spm12):

```bash
pip install pybnt[external]
```

Dev dependencies (pytest, black, ruff):

```bash
pip install pybnt[dev]
```

### Platform Support

PyBrainViewer runs on Windows, macOS, and Linux. FreeSurfer integration requires macOS or Linux. Elastix integration requires a platform-specific binary or the `pyelastix` Python package.

## Quick Start

### GUI Mode (Recommended for interactive use)

```bash
# Launch the main window
pybnt-gui
# Or equivalently
python -m pybnt
```

The GUI opens with a file browser panel on the left (surface, node, edge, and image inputs), a processing options panel with checkboxes, a progress bar, and a log output window. Use the toolbar or File menu to load your data, then click "Visualize" for 3D rendering. The Process menu gives access to connectivity computation, image registration, and AI super-resolution.

### TUI Mode (Terminal-based browsing)

```bash
python -m pybnt.tui.app
```

The Textual-based TUI provides a file browser tree, file-type aware processing when a file is selected (parsing .nv, .node, .edge, .nii, .dcm files), and one-click buttons for surface generation, connectivity, registration, super-resolution, segmentation, and denoising.

### CLI Mode (Scripting and automation)

The CLI entry point is registered as `pybnt`:

```bash
# Show help
pybnt --help
```

Note: The CLI module is under active development. Current commands include surface visualization, node and edge rendering, connectivity analysis, image registration, super-resolution, segmentation, and the LLM assistant. Run `pybnt --help` for the latest list of subcommands.

### Python API

```python
import pybnt

# Parse a BrainNet Viewer .nv surface file
nodes, tris = pybnt.core.surface.parse_surface("BrainMesh_ICBM152.nv")
print(f"Surface: {len(nodes)} vertices, {len(tris)} triangles")

# Parse a node file (6-column tab-delimited: x y z value size label)
node_list = pybnt.core.nodes.parse_node_file("Node_AAL90.node")
for node in node_list:
    print(f"  {node['label']}: ({node['x']:.1f}, {node['y']:.1f}, {node['z']:.1f})")

# Parse an edge connectivity matrix
edge_matrix = pybnt.core.edges.parse_edge_file("Edge_AAL90.edge")
print(f"Edge matrix shape: {edge_matrix.shape}")

# Compute an fMRI correlation matrix from a 4D NIfTI file
matrix = pybnt.processing.connectivity.compute_correlation_matrix("fmri_4d.nii")

# Map volume data to a brain surface using Gaussian-weighted method
vertex_data = pybnt.core.volume.map_volume_to_surface(
    "stat_map.nii", "BrainMesh_ICBM152.nv", algorithm="gaussian"
)
```

#### 3D Rendering

```python
from pybnt.visualization.plotter import draw_surface, draw_nodes
from pybnt.visualization.layouts import apply_layout, six_view
from pybnt.visualization.lighting import apply_lighting, add_ambient_light

# Simple surface render
plotter = draw_surface("BrainMesh_ICBM152.nv", colormap="white", smooth=True)
apply_layout(plotter, "lateral_left")
plotter.show()

# Nodes with edges and surface overlay
plotter = draw_nodes(
    "Node_AAL90.node",
    edge_path="Edge_AAL90.edge",
    surface_path="BrainMesh_ICBM152.nv",
    node_type="all",
    color_tuple=("red", "blue", "white"),
    proj_type="nearest",
    node_scale=1.0,
)
plotter.show()

# Six-view layout
import pyvista as pv
import numpy as np
from pybnt.core.surface import parse_surface

nodes, tris = parse_surface("BrainMesh_ICBM152.nv")
mesh = pv.make_tri_mesh(np.array(nodes), np.array(tris))

plotter = pv.Plotter(shape=(2, 3))
six_view(plotter, [mesh])
plotter.show()
```

#### Custom Colormaps and Lighting

```python
from pybnt.visualization.colormaps import apply_colormap, load_colormap, BUILTIN_COLORMAPS
from pybnt.visualization.lighting import BUILTIN_LIGHTS

# List available colormaps
print(BUILTIN_COLORMAPS.keys())
# ['jet', 'hot', 'cold', 'spectral', 'red_white_blue', 'green_white_red', 'white', 'custom']

# Apply colormap to scalar data
import numpy as np
scalars = np.random.rand(100)
rgb = apply_colormap(scalars, colormap="hot", vmin=0.0, vmax=1.0)

# Load a custom colormap from a BNV-format text file
cmap = load_colormap("my_colormap.txt")

# List available light configurations
print(BUILTIN_LIGHTS.keys())
# ['default', 'head', 'tail', 'left', 'right', 'none']
```

#### Processing Pipelines

```python
# DICOM to NIfTI conversion
pybnt.core.io.dicom_to_nifti("scan.dcm", "scan.nii")

# Image quality metrics
from pybnt.core.imageproc import snr, cnr, psnr, ssim, rmse
import numpy as np
img = np.random.rand(256, 256).astype(np.float32)
print(f"SNR: {snr(img, 'all'):.2f}")

# Connectivity matrix from multiple input types
from pybnt.processing.connectivity import compute_correlation_matrix

# From 4D NIfTI file
mat = compute_correlation_matrix("fmri_4d.nii")

# From a raw numpy array
data = np.random.rand(50, 64, 64, 64)
mat = compute_correlation_matrix(data)

# From a directory of images
mat = compute_correlation_matrix("/path/to/frames/", input_type="path_dir")

# Create an edge matrix from node pairs
from pybnt.core.edges import make_edge_from_nodes
edges = make_edge_from_nodes(
    size=(90, 90),
    outpath="connectivity.edge",
    pairnodesl=[(0, 1, 0.8), (1, 2, 0.5), (2, 0, 0.3)],
)
```

#### AI Features

```python
# LLM brain science assistant
from pybnt.ai.llm import ask_llm

response = ask_llm(
    message=[{"role": "user", "content": "What does the default mode network do?"}],
    api_key="your-dashscope-api-key",  # or set DASHSCOPE_API_KEY env var
    model="qwen-plus",
)

# Brain segmentation (with Otsu fallback when no model available)
from pybnt.ai.segmentation import predict_segmentation, segment_brain
import numpy as np

volume = np.random.rand(128, 128, 128).astype(np.float32)
mask = predict_segmentation(volume)  # Uses Otsu thresholding by default
print(f"Segmentation mask shape: {mask.shape}, foreground: {mask.sum()} voxels")

# Segment a NIfTI file and save the mask
output_path = segment_brain("brain.nii", "brain_mask.nii.gz")

# Image correction (denoising and motion artifact removal)
from pybnt.ai.correction import correct_image

noisy = np.random.rand(256, 256).astype(np.uint8)
denoised = correct_image(noisy, method="denoise", h=10)

# Super-resolution inference (requires trained EDSR checkpoint)
from pybnt.processing.superres import run_superres

run_superres(
    data_dir="./low_res_input/",
    checkpoint_dir="./models/edsr/",
    output_dir="./super_res_output/",
    scale=2,
)
```

#### FreeSurfer and SPM Integration

```python
# Download FreeSurfer automatically
from pybnt.external.freesurfer import download_freesurfer
freesurfer_dir = download_freesurfer()

# Run recon-all pipeline
from pybnt.external.freesurfer import run_freesurfer
run_freesurfer("T1.nii", subject_id="subject01", output_dir="./output")

# Read and write SPM-compatible NIfTI volumes
from pybnt.external.spm import read_spm_volume, write_spm_volume, glm_analysis
import numpy as np

data = read_spm_volume("structural.nii")
write_spm_volume(data, "output.nii")

# Pure-Python GLM analysis (no MATLAB required)
X = np.random.randn(100, 3)
y = X @ np.array([1.0, -0.5, 0.2]) + 0.1 * np.random.randn(100)
result = glm_analysis(X, y)
print(f"Beta coefficients: {result['betas']}")
print(f"R-squared: {result['r_squared']:.3f}")

# Parse SPM.mat files from existing SPM analyses
from pybnt.external.spm import parse_spm_mat
spm_contents = parse_spm_mat("SPM.mat")
```

## Features

### Supported File Formats

PyBrainViewer reads and writes a wide range of neuroimaging and mesh formats:

| Format | Extension | Direction | Description |
|--------|-----------|-----------|-------------|
| BrainNet Surface | `.nv` | Read | Native BrainNet Viewer text surface format (vertex count, vertices, triangle count, triangles) |
| FreeSurfer Surface | `.pial` | Read | FreeSurfer surface reconstruction geometry (via nibabel freesurfer IO) |
| GIFTI | `.gii` | Read | Neuroimaging GIFTI surface format with vertex and face data arrays |
| Wavefront OBJ | `.obj` | Read | Standard 3D mesh format (v and f lines, handles /vt/vn face syntax) |
| BrainNet Node | `.node` | Read | 6-column tab-delimited file: x, y, z, value, size, label |
| BrainNet Edge | `.edge` | Read/Write | N x N connectivity matrix with space or tab separators |
| NIfTI | `.nii` / `.nii.gz` | Read/Write | Standard neuroimaging volume format |
| DICOM | `.dcm` | Read | Medical imaging format (single-slice) |
| Image | `.png` / `.jpg` / `.tiff` | Read/Write | Standard image formats for export and input |

### Visualization Options

- **Surface rendering**: Wireframe or solid shading, configurable colormap and opacity, Laplacian smoothing, anti-aliasing (MSAA 16x), edge display toggle.
- **Node rendering**: Sphere markers with per-node radius scaling from file, configurable color, automatic size from node file.
- **Edge rendering**: Lines between connected node pairs with configurable thickness, color, and opacity. Directed networks supported through arrow conventions.
- **Volume-to-surface projection**: 9 algorithms — all, nearest, surround (distance-thresholded), minium, maxium, average, and Gaussian-weighted variants of each, plus bilinear interpolation. Optional region-bound filtering.
- **Camera presets**: Lateral left/right, medial left/right, ventral, dorsal, anterior, posterior.
- **Six-view layout**: 2 x 3 grid showing lateral, medial, ventral, dorsal, and posterior views simultaneously.
- **Colormaps**: 7 built-in (jet, hot, cold, spectral, red-white-blue, green-white-red, white) plus custom RGB colormap loading from BNV-format text files.
- **Lighting**: 5 preset light configurations (default, head, tail, left, right) plus configurable ambient light fill.
- **ROI overlay**: Render region-of-interest voxels on top of the surface mesh.
- **Voxel rendering**: 3D uniform grid visualization of volumetric data.
- **Export**: Screenshots via `save_screenshot()`, orbit animations via `save_animation()`, frame extraction and image-to-video conversion tools.

### Processing Pipelines

- **DICOM to NIfTI**: Convert individual DICOM files to NIfTI volume format, with automatic min-max normalization to [0, 255].
- **DICOM to Image**: Extract and normalize pixel data to standard image formats.
- **Image normalization**: Min-max scaling to [0, 255] range.
- **Quality metrics**: SNR (full-image or ROI-based), CNR (full-image or ROI-based), PSNR, SSIM, NRMSE.
- **Geometric transforms**: Rotation, uniform scaling, horizontal flip.
- **2D/3D conversion**: Slice 3D volumes into 2D image lists, or stack 2D slices into 3D volumes.
- **Train/test splitting**: Split image datasets by directory or array with configurable ratio, with optional NIfTI-to-slice conversion.
- **Connectivity analysis**: fMRI correlation matrix computation via nilearn, supporting 4D NIfTI files, directories of images, raw arrays, and nibabel objects. Uses PCC seed region with configurable smoothing.
- **Image registration**: Elastix-based linear registration for pairs of images, plus motion correction for fMRI time series. Automatic elastix download when not found.
- **Mesh curvature**: Principal curvature computation (mean, Gaussian, min) via PyMeshLab.
- **Mesh distance**: Average nearest-neighbor distance between two point clouds via KD-tree.
- **Triangle distortion**: Area and angle distortion ratios under projection.

### AI Features

- **LLM Assistant**: Ask brain science questions through the Dashscope API. Supports configurable model selection (default qwen-plus), API key management via environment variable, and streaming responses.
- **Super-resolution**: EDSR (Enhanced Deep Super-Resolution) network with configurable depth, filters, and scale factor. Training pipeline with MAE/MSE loss, checkpointing, PSNR evaluation, and GPU support. Dual-input Combi variant for multi-modal (e.g., T1 + FLAIR) super-resolution.
- **Brain segmentation**: Deep learning-based brain tissue segmentation with automatic fallback to Otsu thresholding. Supports PyTorch model checkpoints. Falls back through SimpleITK Otsu, then scikit-image Otsu.
- **Image correction**: Non-local means denoising for 2D and 3D images, with OpenCV or scikit-image backends. Motion artifact correction via Gaussian blur (2D) or rolling temporal average (3D). Automatic fallback between backends.

### External Tool Integration

- **FreeSurfer**: Full recon-all pipeline wrapper with environment setup, automatic download of FreeSurfer 8.0.0-beta for Linux and macOS, and hemisphere splitting support.
- **SPM12**: NIfTI volume read/write with automatic SPM12/babel fallback, SPM.mat parsing, MATLAB batch job execution, and a pure-Python/NumPy General Linear Model (OLS) implementation with t-statistics, R-squared, and covariance estimation.
- **Elastix**: Image registration wrapper with automatic binary discovery and parameter file support.

## Architecture

```
pybnt/
├── __init__.py              # Package init, lazy submodule loading, version 0.2.0
├── __main__.py              # Entry point for `python -m pybnt` (launches GUI)
│
├── cli/                     # Click-based command-line interface (in development)
│   └── __init__.py
│
├── tui/                     # Textual-based terminal user interface
│   └── app.py               # BrainViewTUI: file browser, processing buttons, log
│
├── gui/                     # PyQt6 graphical user interface
│   ├── app.py               # Ui_MainWindow: menus, toolbar, file inputs, viz, threading
│   ├── dialogs/             # Dialog widgets
│   └── widgets/             # Custom widgets
│
├── core/                    # Core data structures and algorithms
│   ├── surface.py           # Surface parsing (.nv, .pial, .gii, .obj)
│   ├── nodes.py             # Node file parsing, ROI stats, connectivity values
│   ├── edges.py             # Edge file parsing, matrix construction from pairs
│   ├── projection.py        # Triangle/Surfaces projection classes (9 algorithms)
│   ├── volume.py            # Volume-to-surface mapping (7 algorithms)
│   ├── io.py                # DICOM/NIfTI I/O, format conversion
│   ├── imageproc.py         # Normalization, SNR/CNR/PSNR/SSIM, transforms
│   ├── dataproc.py          # Train/test splitting, NIfTI-to-slice conversion
│   └── utils.py             # Distance metrics, curvature, mesh distance, image loading
│
├── processing/              # Processing pipelines
│   ├── connectivity.py      # fMRI correlation matrix via nilearn
│   ├── superres.py          # EDSR training and inference
│   ├── registration.py      # Elastix registration and motion correction
│   ├── segmentation.py      # FreeSurfer recon-all wrapper
│   └── models/              # ML model definitions
│       ├── edsr.py          # EDSR and Combi model architectures
│       ├── dataset.py       # TF dataset pipeline
│       ├── train.py         # EdsrTrainer with checkpointing
│       ├── common.py        # Shared model utilities
│       └── utils.py         # Training helpers
│
├── visualization/           # 3D rendering engine
│   ├── plotter.py           # draw_surface, draw_nodes, draw_voxels, draw_roi
│   ├── colormaps.py         # 7 built-in colormaps + custom loader
│   ├── layouts.py           # Camera presets, six-view layout
│   ├── lighting.py          # 5 light configs + ambient light
│   └── export.py            # Screenshot, video, frame extraction
│
├── ai/                      # AI-powered features
│   ├── llm.py               # Dashscope LLM assistant (Qwen models)
│   ├── segmentation.py      # AI brain segmentation with Otsu fallback
│   └── correction.py        # Image denoising and motion correction
│
├── external/                # External tool wrappers
│   ├── freesurfer.py        # FreeSurfer download and recon-all wrapper
│   ├── spm.py               # SPM12 I/O, batch runner, GLM analysis
│   └── elastix.py           # Elastix registration binary wrapper
│
└── data/                    # Surface template files
    └── templates/           # 21 .nv surface templates + 7 atlas label sets
        ├── BrainMesh_ICBM152.nv             # Standard ICBM152 template
        ├── BrainMesh_ICBM152_smoothed.nv    # Smoothed variant
        ├── BrainMesh_ICBM152_tal.nv         # Talairach version
        ├── BrainMesh_Ch2.nv                 # Ch2 template
        ├── BrainMesh_Cerebellum.nv           # Cerebellum template
        ├── BrainMesh_ICBM152Left.nv          # Left hemisphere
        ├── BrainMesh_ICBM152Right.nv         # Right hemisphere
        ├── AAL90/                            # Automated Anatomical Labeling atlas
        ├── Power264/                         # Power 264-node atlas
        ├── LPBA40/                           # LPBA40 atlas
        ├── HOA112/                           # Hammersmith atlas
        ├── Brodmann82/                       # Brodmann areas atlas
        ├── Desikan-Killiany68/               # Desikan-Killiany atlas
        ├── Dos160/                           # DOS 160 atlas
        └── Fair34/                           # Fair 34-node atlas
```

### Module Dependency Flow

```
GUI / TUI / CLI (entry points)
    │
    ├─── visualization/ ─── core/ ─── processing/
    │       │                │            │
    │       │           surface.py    connectivity.py
    │       │           nodes.py      superres.py
    │       │           edges.py      registration.py
    │       │           projection.py segmentation.py
    │       │           volume.py
    │       │           io.py
    │       │           imageproc.py
    │       │           utils.py
    │       │
    │       ├─── colormaps.py
    │       ├─── layouts.py
    │       ├─── lighting.py
    │       └─── export.py
    │
    ├─── ai/ ─── core/imageproc.py
    │       │
    │       ├─── llm.py           (external: dashscope API)
    │       ├─── segmentation.py  (core: io, utils)
    │       └─── correction.py    (core: imageproc)
    │
    └─── external/
            ├─── freesurfer.py    (subprocess → FreeSurfer binary)
            ├─── spm.py           (nibabel, optional spm12 MATLAB bridge)
            └─── elastix.py       (subprocess → elastix binary)
```

## BNV Compatibility

PyBrainViewer is designed as a drop-in Python replacement for BrainNet Viewer (BNV), the widely-used MATLAB toolbox by Mingrui Xia. File format compatibility is a priority:

### Surface Files (.nv)

The native BrainNet Viewer surface format uses a simple text structure:

```
<nodenum>
x1 y1 z1
x2 y2 z2
...
<trinum>
v1a v1b v1c
...
```

PyBrainViewer parses this format directly with the same vertex/face indexing convention. The 21 surface templates shipped with BNV (ICBM152, Ch2, cerebellum, hemispheric splits, and all smoothing/Talairach variants) are included in `pybnt/data/templates/` and can be referenced by filename without additional downloads.

### Node Files (.node)

The 6-column tab-delimited node format is fully supported:

| Column | Field | Type | Description |
|--------|-------|------|-------------|
| 1 | x | float | X coordinate (MNI/Talairach) |
| 2 | y | float | Y coordinate |
| 3 | z | float | Z coordinate |
| 4 | value | float | Node value for colormap mapping |
| 5 | size | float | Node radius for sphere rendering |
| 6 | label | string | Region name or label text |

Tolerance for malformed lines is configurable via the `mode` parameter (`"continue"` skips errors, any other value aborts on first error).

### Edge Files (.edge)

The N x N connectivity matrix format is fully supported. Values can be space-separated or tab-separated. Utility functions for constructing edge matrices from node-pair lists are provided, supporting both unweighted (binary) and weighted connections.

### Atlas Labels

Seven atlas label sets from BNV are included:

- **AAL90** — Automated Anatomical Labeling, 90 regions
- **Power264** — Power 264-node functional atlas
- **LPBA40** — LONI Probabilistic Brain Atlas, 40 regions
- **HOA112** — Hammersmith atlas, 112 regions
- **Brodmann82** — Brodmann cytoarchitectonic areas, 82 regions
- **Desikan-Killiany68** — FreeSurfer cortical atlas, 68 regions
- **Dos160** — Dosenbach 160-node atlas
- **Fair34** — Fair 34-node atlas

### Colormap Files

Custom colormaps in BNV format (one RGB triplet per line, space-separated integers 0-255) are fully supported via `load_colormap()`.

## API Reference

### Core Module (`pybnt.core`)

| Function | Description |
|----------|-------------|
| `surface.parse_surface(path)` | Parse .nv, .pial, .gii, or .obj files → (nodes, triangles) |
| `nodes.parse_node_file(path, mode)` | Parse .node text file → list of node dicts |
| `nodes.roi_compute(roinodes, scalfac, mode)` | ROI summary: average, max, or min |
| `nodes.node2connect(nodelist, matrix)` | Node values from connectivity matrix |
| `nodes.calculate_node_value_from_image(image, mask, stat)` | Node values from image data |
| `edges.parse_edge_file(path, mode)` | Parse .edge matrix → 2D numpy array |
| `edges.make_edge_from_nodes(size, outpath, pairs)` | Build edge matrix from node pairs |
| `io.dicom_to_nifti(dpath, npath)` | Convert DICOM → NIfTI |
| `io.dicom_to_image(dpath, ipath)` | Convert DICOM → image |
| `io.nifti_to_image(npath, ipath, normalize)` | Convert NIfTI → image |
| `io.image_to_nifti(img, npath)` | Convert image → NIfTI |
| `io.nifti_shape(npath)` | Get NIfTI volume shape |
| `imageproc.normalize(img)` | Min-max normalize to [0, 255] |
| `imageproc.snr(img, type)` | Signal-to-noise ratio |
| `imageproc.cnr(img, type)` | Contrast-to-noise ratio |
| `imageproc.psnr(img1, img2)` | Peak SNR between two images |
| `imageproc.ssim(img1, img2)` | Structural similarity index |
| `imageproc.rmse(img1, img2)` | Normalized root mean square error |
| `imageproc.transform(img, angle, scale, flip)` | Geometric transform |
| `imageproc.to_2d(img3d, axis)` | Split 3D volume into 2D slices |
| `imageproc.to_3d(slices, axis)` | Stack 2D slices into 3D volume |
| `projection.Triangle` | Single triangle with projection math |
| `projection.Surfaces` | Collection of triangles with batch projection |
| `dataproc.split_train_test(imgpath, train, test, ...)` | Split dataset into train/test |
| `utils.curvature(mesh, path)` | Mesh principal curvatures |
| `utils.avrdistance(mesh1, mesh2)` | Average mesh-to-mesh distance |
| `utils.readimage(path)` | Load image (NIfTI or standard format) |

### Visualization Module (`pybnt.visualization`)

| Function | Description |
|----------|-------------|
| `plotter.draw_surface(path, colormap, smooth, light, opacity)` | Render brain surface mesh |
| `plotter.draw_nodes(node_path, edge_path, surface_path, ...)` | Render nodes with optional edges and surface |
| `plotter.draw_voxels(array)` | Render 3D voxel grid |
| `plotter.draw_roi(image, mask)` | Render region of interest |
| `plotter.draw_roi_with_surface(surface, image, mask)` | ROI overlay on surface |
| `colormaps.load_colormap(path)` | Load BNV-format colormap file |
| `colormaps.apply_colormap(scalars, colormap, vmin, vmax)` | Apply colormap to scalars |
| `layouts.apply_layout(plotter, layout_name)` | Set camera to preset view |
| `layouts.six_view(plotter, surfaces, nodes, edges)` | 6-view subplot layout |
| `layouts.medial_view(plotter)` | Set medial camera view |
| `lighting.apply_lighting(plotter, config)` | Apply lighting configuration |
| `lighting.add_ambient_light(plotter, intensity)` | Add ambient light fill |
| `export.save_screenshot(plotter, path)` | Save screenshot to file |
| `export.save_animation(plotter, path, n_frames)` | Save orbit animation |
| `export.images_to_video(dir, output, fps)` | Create video from image sequence |

### Processing Module (`pybnt.processing`)

| Function | Description |
|----------|-------------|
| `connectivity.compute_correlation_matrix(input, type)` | fMRI correlation matrix |
| `connectivity.compute_connectivity_from_directory(dir)` | Connectivity from image directory |
| `superres.train_superres(img_dir, save_dir, label_dir, ...)` | Train EDSR super-resolution model |
| `superres.run_superres(data_dir, checkpoint, output, ...)` | Run super-resolution inference |
| `superres.superres_single_image(image, model, ...)` | Single-image super-resolution |
| `registration.register_image(fixed, moving, output)` | Register images via elastix |
| `registration.correct_motion(image_dir, ref_idx, output)` | Motion correction for fMRI |
| `segmentation.segment_brain(input, subject_id, output)` | Run FreeSurfer recon-all |
| `segmentation.download_freesurfer(target_dir)` | Download FreeSurfer 8.0.0-beta |

### AI Module (`pybnt.ai`)

| Function | Description |
|----------|-------------|
| `llm.ask_llm(messages, api_key, model)` | Ask brain science question to LLM |
| `segmentation.predict_segmentation(image, model_path)` | Predict brain segmentation mask |
| `segmentation.segment_brain(file_path, output_path)` | Segment NIfTI file and save |
| `correction.correct_image(image, method, ...)` | Apply image correction/denoising |
| `correction.train_correction_model(data_dir, save_dir)` | Train correction model |

### External Module (`pybnt.external`)

| Function | Description |
|----------|-------------|
| `freesurfer.run_freesurfer(input, subject, output)` | Run FreeSurfer recon-all pipeline |
| `freesurfer.download_freesurfer(target_dir)` | Download FreeSurfer installer |
| `spm.read_spm_volume(path)` | Read NIfTI via SPM12 or nibabel |
| `spm.write_spm_volume(data, path, affine)` | Write NIfTI volume |
| `spm.parse_spm_mat(path)` | Parse SPM.mat file into dict |
| `spm.run_spm_batch(script, matlab_cmd, timeout)` | Run SPM batch via MATLAB |
| `spm.glm_analysis(design_matrix, data)` | Pure-Python GLM (OLS) analysis |
| `elastix.register_elastix(fixed, moving, output, ...)` | Register images via elastix binary |

## Contributing

Contributions are welcome. The project is in early alpha (v0.2.0) and there is plenty of room for improvement. Areas especially in need of attention:

- **CLI implementation**: The Click-based CLI is registered but not yet populated with subcommands. See `pybnt/cli/__init__.py`.
- **Tests**: No test suite exists yet. Unit tests for core parsing, projection, and processing functions would be highly valuable.
- **Documentation**: Additional examples and tutorials, particularly for the AI segmentation and correction pipelines.
- **Formats**: Support for additional surface formats (e.g., STL, PLY) and edge formats.

### Development Setup

```bash
git clone https://github.com/tzhazuma/PyBNT.git
cd PyBNT
pip install -e ".[dev]"
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **BrainNet Viewer** (Mingrui Xia, Beijing Normal University) — The original MATLAB brain network visualization and analysis tool that inspired this project and whose file format compatibility we maintain.
- **NiBabel** — The foundational Python library for neuroimaging file format I/O (NIfTI, GIFTI, FreeSurfer).
- **PyVista** — 3D visualization and mesh analysis library that powers all rendering in PyBrainViewer.
- **Nilearn** — Statistical learning for neuroimaging, used for fMRI connectivity analysis.
- **PyQt6** — Cross-platform GUI framework for the desktop application.
- **Textual** — Terminal user interface framework for the TUI mode.
- **FreeSurfer** (Martinos Center, MGH) — Brain segmentation and surface reconstruction toolkit.
- **SPM12** (Wellcome Centre, UCL) — Statistical Parametric Mapping software.
- **Elastix** — Image registration toolbox.
- **EDSR** (Enhanced Deep Super-Resolution) — The deep learning architecture used for brain image super-resolution.

## Citation

If you use PyBrainViewer in your research, please cite:

```
Tang, Z. (2025). PyBrainViewer: AI-Powered Multi-Function Brain Visual Tool.
ShanghaiTech University. https://github.com/tzhazuma/PyBNT
```

## Version History

- **v0.2.0** (Current) — Core surface/node/edge parsing, PyQt6 GUI, Textual TUI, 3D rendering with PyVista, EDSR super-resolution, LLM assistant, FreeSurfer/SPM/Elastix wrappers, connectivity analysis, image processing, 21 surface templates, 8 atlas label sets.
- **v0.1.0** — Initial prototype with basic BNV file parsing and MATLAB BrainNet Viewer compatibility layer.

---

*Maintained by Tang Zhihao, ShanghaiTech University.*
