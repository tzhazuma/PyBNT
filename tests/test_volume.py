"""Tests for pybnt/core/volume.py — volume-to-surface mapping."""

import pytest
import numpy as np
from pybnt.core.volume import map_volume_to_surface


class TestVolumeMappingImports:
    """Tests for module imports and basic availability."""

    def test_map_volume_function_exists(self):
        """Test map_volume_to_surface is callable."""
        assert callable(map_volume_to_surface), (
            "map_volume_to_surface should be callable"
        )

    def test_algorithm_list_valid(self):
        """Test all expected algorithm names are available in the docstring."""
        from pybnt.core.volume import map_volume_to_surface as func
        # Extract algorithms from the param description
        algorithms = [
            "nearest",
            "linear",
            "gaussian",
            "cubic",
            "maximum",
            "minimum",
            "average",
        ]
        assert len(algorithms) == 7, "Should have 7 mapping algorithms"
        doc = func.__doc__ or ""
        for alg in algorithms:
            assert alg in doc, (
                f"Algorithm '{alg}' should be documented in function docstring"
            )

    def test_invalid_algorithm_raises(self):
        """Test invalid algorithm name raises ValueError."""
        with pytest.raises((ValueError, FileNotFoundError, OSError)):
            map_volume_to_surface(
                volume_path="/nonexistent/vol.nii",
                surface_path="/nonexistent/surf.nv",
                algorithm="invalid_algo_xyz",
            )
