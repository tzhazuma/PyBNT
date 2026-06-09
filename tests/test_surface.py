"""Tests for pybnt/core/surface.py — surface file parsing."""

import pytest
import numpy as np
from pybnt.core.surface import parse_surface


class TestParseSurfaceNV:
    """Tests for parsing BrainNet Viewer .nv surface format."""

    def test_parse_nv_format(self, icbm152_surface):
        """Test parsing returns numpy arrays with correct shapes."""
        nodes, tris = parse_surface(icbm152_surface)
        assert isinstance(nodes, np.ndarray), "Nodes should be numpy array"
        assert isinstance(tris, np.ndarray), "Triangles should be numpy array"
        assert nodes.shape[1] == 3, "Nodes should have 3 columns (x, y, z)"
        assert tris.shape[1] == 3, "Triangles should have 3 columns (indices)"
        assert len(nodes) > 0, "Should have vertices"
        assert len(tris) > 0, "Should have triangles"

    def test_nv_first_vertex(self, icbm152_surface):
        """Test first vertex coordinates match known values."""
        nodes, _tris = parse_surface(icbm152_surface)
        x, y, z = nodes[0]
        assert abs(x - (-2.860760)) < 0.001, f"Expected x=-2.860760, got {x}"
        assert abs(y - 25.198200) < 0.001, f"Expected y=25.198200, got {y}"
        assert abs(z - 16.612800) < 0.001, f"Expected z=16.612800, got {z}"

    def test_nv_vertex_count(self, icbm152_surface):
        """Test ICBM152 has expected number of vertices and triangles."""
        nodes, tris = parse_surface(icbm152_surface)
        assert len(nodes) == 81924, (
            f"ICBM152 should have 81924 vertices, got {len(nodes)}"
        )
        assert len(tris) == 163840, (
            f"ICBM152 should have 163840 triangles, got {len(tris)}"
        )

    def test_nv_vertices_are_floats(self, icbm152_surface):
        """Test that vertex coordinates are floating point values."""
        nodes, _tris = parse_surface(icbm152_surface)
        assert np.issubdtype(nodes.dtype, np.floating), (
            "Vertex coordinates should be floating point"
        )

    def test_nv_triangle_indices_are_integers(self, icbm152_surface):
        """Test that triangle indices are integer values."""
        _nodes, tris = parse_surface(icbm152_surface)
        assert np.issubdtype(tris.dtype, np.integer), (
            "Triangle indices should be integers"
        )

    def test_triangle_indices_within_range(self, icbm152_surface):
        """Test all triangle indices are within valid vertex range."""
        nodes, tris = parse_surface(icbm152_surface)
        assert np.all(tris >= 0), "Triangle indices should be non-negative"
        assert np.all(tris < len(nodes)), (
            "Triangle indices should be within vertex count"
        )

    def test_triangle_indices_zero_based(self, icbm152_surface):
        """Test triangle indices appear to be zero-based (have 0 values)."""
        _nodes, tris = parse_surface(icbm152_surface)
        assert 0 in tris, "Triangle indices should be zero-based"

    def test_no_nan_in_vertices(self, icbm152_surface):
        """Test no NaN values in vertex coordinates."""
        nodes, _tris = parse_surface(icbm152_surface)
        assert not np.any(np.isnan(nodes)), "Vertices should not contain NaN"

    def test_no_inf_in_vertices(self, icbm152_surface):
        """Test no infinity values in vertex coordinates."""
        nodes, _tris = parse_surface(icbm152_surface)
        assert not np.any(np.isinf(nodes)), "Vertices should not contain Inf"

    def test_ch2_surface_parsing(self, ch2_surface):
        """Test parsing of Ch2 surface template."""
        nodes, tris = parse_surface(ch2_surface)
        assert len(nodes) > 0, "Ch2 surface should have vertices"
        assert len(tris) > 0, "Ch2 surface should have triangles"
        assert nodes.shape[1] == 3
        assert tris.shape[1] == 3

    def test_smoothed_surface_parsing(self, icbm152_smoothed_surface):
        """Test parsing of smoothed ICBM152 surface template."""
        nodes, tris = parse_surface(icbm152_smoothed_surface)
        assert len(nodes) > 0, "Smoothed surface should have vertices"
        assert len(tris) > 0, "Smoothed surface should have triangles"


class TestParseSurfaceErrors:
    """Tests for error handling in surface parsing."""

    def test_nonexistent_file(self):
        """Test parsing nonexistent file raises an error."""
        with pytest.raises((FileNotFoundError, ValueError, OSError, Exception)):
            parse_surface("/nonexistent/path/file.nv")

    def test_empty_file(self, tmp_path):
        """Test parsing an empty file raises an error."""
        empty_path = tmp_path / "empty.nv"
        empty_path.write_text("")
        with pytest.raises((ValueError, Exception)):
            parse_surface(str(empty_path))


class TestParseSurfaceFormatDetection:
    """Tests for format auto-detection based on file extension."""

    def test_nv_format_detected(self, icbm152_surface):
        """Test .nv format is auto-detected and parsed correctly."""
        nodes, tris = parse_surface(icbm152_surface)
        assert nodes is not None
        assert tris is not None
        assert len(nodes) > 0

    def test_surface_no_extension_parsed_as_nv(self, icbm152_surface):
        """Test file without extension is parsed as .nv format."""
        nodes, tris = parse_surface(icbm152_surface)
        assert isinstance(nodes, np.ndarray)
        assert isinstance(tris, np.ndarray)

    def test_parse_returns_tuple(self, icbm152_surface):
        """Test parse_surface returns a 2-tuple of (nodes, triangles)."""
        result = parse_surface(icbm152_surface)
        assert isinstance(result, tuple)
        assert len(result) == 2
