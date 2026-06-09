"""Lighting configurations for brain rendering."""

from dataclasses import dataclass


@dataclass
class LightConfig:
    position: tuple = (-20, -20, 0)
    focal_point: tuple = (0, 0, 0)
    intensity: float = 0.5
    color: str = "white"


BUILTIN_LIGHTS = {
    "default": LightConfig((-20, -20, 0), (0, 0, 0), 0.5),
    "head": LightConfig((0, 0, 1), (0, 0, 0), 0.5),
    "tail": LightConfig((0, 0, -1), (0, 0, 0), 0.5),
    "left": LightConfig((0, 1, 0), (0, 0, 0), 0.5),
    "right": LightConfig((0, -1, 0), (0, 0, 0), 0.5),
    "none": None,
}


def apply_lighting(plotter, config):
    """Apply lighting configuration to plotter.

    Args:
        plotter: pyvista.Plotter instance.
        config: LightConfig, str key into BUILTIN_LIGHTS, or None.
    """
    if config is None:
        return
    if isinstance(config, str):
        normalized = config.lower()
        if normalized == "none":
            return
        if normalized not in BUILTIN_LIGHTS:
            raise ValueError(f"Unknown light config: {config}")
        config = BUILTIN_LIGHTS[normalized]
    if config is None:
        return

    import pyvista as pv
    light = pv.Light(
        position=config.position,
        focal_point=config.focal_point,
        intensity=config.intensity,
        color=config.color,
    )
    plotter.add_light(light)


def add_custom_light(plotter, position, focal_point, intensity=0.5, color="white"):
    """Add a user-defined directional light to the plotter.

    Args:
        plotter: pyvista.Plotter instance.
        position: (x, y, z) tuple for the light source position.
        focal_point: (x, y, z) tuple that the light points toward.
        intensity: Brightness of the light (default 0.5).
        color: Light color as a name string or RGB tuple.
    """
    import pyvista as pv

    light = pv.Light(
        position=position,
        focal_point=focal_point,
        intensity=intensity,
        color=color,
    )
    plotter.add_light(light)


def add_ambient_light(plotter, intensity=0.3):
    """Add ambient light fill to plotter.

    Args:
        plotter: pyvista.Plotter instance.
        intensity: Ambient light intensity (default 0.3).
    """
    import pyvista as pv
    plotter.add_light(pv.Light(light_type="ambient", intensity=intensity))


def ambient_light(plotter, intensity=0.3):
    """Add ambient light fill to plotter (BNV-compatible alias).

    Equivalent to ``add_ambient_light``.

    Args:
        plotter: pyvista.Plotter instance.
        intensity: Ambient light intensity (default 0.3).
    """
    add_ambient_light(plotter, intensity=intensity)
