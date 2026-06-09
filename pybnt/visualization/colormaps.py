"""Built-in colormaps for brain visualization with BNV parity."""

import numpy as np

BUILTIN_COLORMAPS = {
    "jet": [(0, 0, 143), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)],
    "hsv": [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)],
    "hot": [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)],
    "cold": [(0, 255, 255), (0, 0, 255), (0, 0, 128)],
    "summer": [(0, 128, 102), (0, 255, 102), (255, 255, 102)],
    "winter": [(0, 0, 255), (0, 128, 255), (0, 255, 128)],
    "spring": [(255, 0, 255), (255, 255, 0)],
    "autumn": [(255, 0, 0), (255, 255, 0), (128, 64, 0)],
    "spectral": [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)],
    "cool": [(0, 255, 255), (255, 0, 255)],
    "warm": [(255, 255, 204), (255, 128, 0), (128, 0, 0)],
    "bone": [(0, 0, 0), (64, 64, 96), (128, 128, 192), (192, 192, 224), (255, 255, 255)],
    "copper": [(0, 0, 0), (191, 128, 64), (255, 199, 127)],
    "pink": [(40, 30, 30), (255, 160, 160), (255, 205, 210), (255, 240, 245)],
    "gray": [(0, 0, 0), (128, 128, 128), (255, 255, 255)],
    "red_white_blue": [(255, 0, 0), (255, 255, 255), (0, 0, 255)],
    "green_white_red": [(0, 255, 0), (255, 255, 255), (255, 0, 0)],
    "white": [(255, 255, 255)],
    "custom": None,
}

# Global range lock for colormap normalization (BNV feature)
_locked_vmin = None
_locked_vmax = None


def load_colormap(path):
    """Load custom colormap from a text file (BNV format).

    Each line should contain three integers: R G B.

    Returns:
        list of (R, G, B) tuples.
    """
    colormap = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                colormap.append(tuple(int(p) for p in parts))
    return colormap


def load_custom_colormap(path):
    """Load an Nx3 colormap from a text file (BNV-compatible).

    Each line should contain three space-separated integers: R G B.
    The loaded colormap is registered under the "custom" key in
    BUILTIN_COLORMAPS for use with ``apply_colormap``.

    Args:
        path: Path to a text file with one ``R G B`` line per entry.

    Returns:
        list of (R, G, B) tuples.
    """
    cmap = load_colormap(path)
    BUILTIN_COLORMAPS["custom"] = cmap
    return cmap


def apply_colormap(scalars, colormap="jet", vmin=None, vmax=None):
    """Apply colormap to scalar data.

    Args:
        scalars: numpy array of scalar values.
        colormap: str key into BUILTIN_COLORMAPS or a list of RGB tuples.
        vmin: lower clamp value (defaults to min(scalars)).
        vmax: upper clamp value (defaults to max(scalars)).

    Returns:
        numpy array of shape (len(scalars), 3) with RGB values in 0-255.
    """
    if isinstance(colormap, str):
        cmap = BUILTIN_COLORMAPS.get(colormap)
        if cmap is None:
            raise ValueError(f"Unknown colormap: {colormap}")
    else:
        cmap = colormap

    cmap = np.array(cmap, dtype=np.float64)

    effective_vmin = _locked_vmin if _locked_vmin is not None else vmin
    effective_vmax = _locked_vmax if _locked_vmax is not None else vmax

    if effective_vmin is None:
        effective_vmin = np.min(scalars)
    if effective_vmax is None:
        effective_vmax = np.max(scalars)

    norm = (scalars - effective_vmin) / (effective_vmax - effective_vmin)
    norm = np.clip(norm, 0.0, 1.0)

    n_colors = len(cmap)
    indices = norm * (n_colors - 1)
    lower_idx = np.clip(np.floor(indices).astype(int), 0, n_colors - 1)
    upper_idx = np.clip(np.ceil(indices).astype(int), 0, n_colors - 1)
    frac = (indices - lower_idx).reshape(-1, 1)

    colors = cmap[lower_idx] * (1.0 - frac) + cmap[upper_idx] * frac
    return colors


def fix_colormap_range(vmin, vmax):
    """Lock the colormap normalization range (BNV feature).

    When set, all subsequent ``apply_colormap`` calls will use these
    fixed bounds instead of computing min/max from the data.

    Call ``fix_colormap_range(None, None)`` to clear the lock.

    Args:
        vmin: Fixed lower bound for normalization.
        vmax: Fixed upper bound for normalization.
    """
    global _locked_vmin, _locked_vmax
    _locked_vmin = vmin
    _locked_vmax = vmax


class ColorBar:
    """Configurable scalar bar for pyvista plots.

    Provides BNV-style control over position, size, labels, and range.

    Args:
        title: Title string displayed next to the bar.
        n_labels: Number of tick labels.
        position_x: Horizontal position (0.0-1.0 fraction of window).
        position_y: Vertical position (0.0-1.0 fraction of window).
        width: Width of the bar.
        height: Height of the bar.
    """

    def __init__(self, title="", n_labels=5, position_x=0.8, position_y=0.05,
                 width=0.08, height=0.5):
        self.title = title
        self.n_labels = n_labels
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height

    def add_to_plotter(self, plotter, vmin=None, vmax=None, colormap="jet"):
        """Attach this scalar bar to a pyvista Plotter.

        Args:
            plotter: pyvista.Plotter instance.
            vmin: Data lower bound for the bar labels.
            vmax: Data upper bound for the bar labels.
            colormap: Colormap name key from BUILTIN_COLORMAPS.
        """
        kwargs = {
            "title": self.title,
            "n_labels": self.n_labels,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "width": self.width,
            "height": self.height,
        }
        if vmin is not None:
            kwargs["fmt"] = "%.2f"
        plotter.add_scalar_bar(**kwargs)


def add_colorbar(plotter, colormap, title="", vmin=None, vmax=None):
    """Add a scalar colorbar to a pyvista Plotter.

    Args:
        plotter: pyvista.Plotter instance.
        colormap: str key into BUILTIN_COLORMAPS.
        title: Label text for the colorbar.
        vmin: Data lower bound (for tick labels).
        vmax: Data upper bound (for tick labels).
    """
    bar = ColorBar(title=title)
    bar.add_to_plotter(plotter, vmin=vmin, vmax=vmax, colormap=colormap)
