"""Tests for pybnt/core/nodes.py — node file parsing and computation."""

import pytest
import numpy as np
from pybnt.core.nodes import (
    parse_node_file,
    calculate_node_pos,
    calculate_node_value,
    calculate_node_size,
    roi_compute,
    calculate_node_value_from_image,
    node2connect,
)


class TestParseNodeFile:
    """Tests for parse_node_file function."""

    def test_parse_aal90_node_file(self, aal90_node):
        """Test parsing AAL90 node file returns 90 nodes."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None, "Should return parsed nodes"
        assert len(nodes) == 90, (
            f"AAL90 should have 90 nodes, got {len(nodes)}"
        )

    def test_node_dict_structure(self, aal90_node):
        """Test each node dict has all required keys."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        node = nodes[0]
        assert "x" in node, "Node should have x coordinate"
        assert "y" in node, "Node should have y coordinate"
        assert "z" in node, "Node should have z coordinate"
        assert "value" in node, "Node should have value"
        assert "size" in node, "Node should have size"
        assert "label" in node, "Node should have label"

    def test_node_coordinate_types(self, aal90_node):
        """Test node coordinates are floating point."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        for node in nodes:
            assert isinstance(node["x"], float), f"x should be float, got {type(node['x'])}"
            assert isinstance(node["y"], float), f"y should be float, got {type(node['y'])}"
            assert isinstance(node["z"], float), f"z should be float, got {type(node['z'])}"

    def test_node_coordinates_in_mni_range(self, aal90_node):
        """Test all node coordinates are within plausible MNI range."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        for node in nodes:
            assert -120 < node["x"] < 120, f"x={node['x']} outside MNI range"
            assert -120 < node["y"] < 120, f"y={node['y']} outside MNI range"
            assert -100 < node["z"] < 120, f"z={node['z']} outside MNI range"

    def test_node_values_are_floats(self, aal90_node):
        """Test node value and size fields are floats."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        for node in nodes:
            assert isinstance(node["value"], float), "value should be float"
            assert isinstance(node["size"], float), "size should be float"

    def test_node_labels_are_strings(self, aal90_node):
        """Test node label field is a string (not empty)."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        labels = [node["label"] for node in nodes]
        assert all(isinstance(lbl, str) for lbl in labels), (
            "All labels should be strings"
        )
        assert all(lbl != "" for lbl in labels), "Labels should not be empty"

    def test_parse_with_continue_mode(self, aal90_node):
        """Test parsing with default 'continue' error mode."""
        nodes = parse_node_file(aal90_node, mode="continue")
        assert nodes is not None

    def test_parse_with_strict_mode(self, aal90_node):
        """Test strict mode returns None when comment/header lines fail parsing."""
        nodes = parse_node_file(aal90_node, mode="strict")
        assert nodes is None, (
            "Strict mode should return None due to comment header line"
        )

    def test_power264_node_file(self, power264_node):
        """Test parsing Power264 node file."""
        nodes = parse_node_file(power264_node)
        assert nodes is not None
        assert len(nodes) == 264, (
            f"Power264 should have 264 nodes, got {len(nodes)}"
        )

    def test_node_values_positive(self, aal90_node):
        """Test all node values are non-negative."""
        nodes = parse_node_file(aal90_node)
        assert nodes is not None
        for node in nodes:
            assert node["value"] >= 0, f"Node value should be >= 0, got {node['value']}"


class TestNodeCalculations:
    """Tests for node position, value, and size calculations."""

    def test_calculate_node_pos_two_nodes(self):
        """Test mean position of two nodes."""
        nodes = [
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 2.0),
        ]
        pos = calculate_node_pos(nodes)
        assert abs(pos[0] - 1.0) < 0.001, f"Expected x=1.0, got {pos[0]}"
        assert abs(pos[1] - 1.0) < 0.001, f"Expected y=1.0, got {pos[1]}"
        assert abs(pos[2] - 1.0) < 0.001, f"Expected z=1.0, got {pos[2]}"

    def test_calculate_node_pos_single_node(self):
        """Test mean position of a single node equals the node position."""
        nodes = [(5.0, -3.0, 10.0)]
        pos = calculate_node_pos(nodes)
        assert abs(pos[0] - 5.0) < 0.001
        assert abs(pos[1] - (-3.0)) < 0.001
        assert abs(pos[2] - 10.0) < 0.001

    def test_calculate_node_pos_with_values(self):
        """Test calculate_node_pos ignores the 4th value column."""
        nodes = [
            (1.0, 2.0, 3.0, 100.0),
            (5.0, 6.0, 7.0, 200.0),
        ]
        pos = calculate_node_pos(nodes)
        assert abs(pos[0] - 3.0) < 0.001
        assert abs(pos[1] - 4.0) < 0.001
        assert abs(pos[2] - 5.0) < 0.001

    def test_calculate_node_value(self):
        """Test mean value calculation."""
        nodes = [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 3.0),
        ]
        val = calculate_node_value(nodes)
        assert abs(val - 2.0) < 0.001, f"Expected 2.0, got {val}"

    def test_calculate_node_value_all_equal(self):
        """Test mean value when all values are equal."""
        nodes = [
            (0.0, 0.0, 0.0, 7.0),
            (0.0, 0.0, 0.0, 7.0),
            (0.0, 0.0, 0.0, 7.0),
        ]
        val = calculate_node_value(nodes)
        assert abs(val - 7.0) < 0.001

    def test_calculate_node_size(self):
        """Test node size calculation with default scaling factor."""
        nodes = [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0, 3.0),
        ]
        size = calculate_node_size(nodes)
        assert abs(size - 0.03) < 0.0001, f"Expected 0.03 (3 * 0.01), got {size}"

    def test_calculate_node_size_custom_fac(self):
        """Test node size with custom scaling factor."""
        nodes = [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 2.0),
        ]
        size = calculate_node_size(nodes, fac=0.5)
        assert abs(size - 1.0) < 0.001, f"Expected 1.0 (2 * 0.5), got {size}"

    def test_calculate_node_size_empty(self):
        """Test node size for empty list returns 0."""
        size = calculate_node_size([])
        assert size == 0.0


class TestROICompute:
    """Tests for roi_compute function."""

    def test_roi_average_mode(self):
        """Test ROI average computation."""
        roinodes = [
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0, 4.0),
        ]
        result = roi_compute(roinodes, mode="average")
        assert abs(result - 3.0) < 0.001

    def test_roi_max_mode(self):
        """Test ROI maximum computation."""
        roinodes = [
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0, 5.0),
            (0.0, 0.0, 0.0, 3.0),
        ]
        result = roi_compute(roinodes, mode="max")
        assert abs(result - 5.0) < 0.001

    def test_roi_min_mode(self):
        """Test ROI minimum computation."""
        roinodes = [
            (0.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0, 5.0),
            (0.0, 0.0, 0.0, 3.0),
        ]
        result = roi_compute(roinodes, mode="min")
        assert abs(result - 2.0) < 0.001

    def test_roi_with_scale_factor(self):
        """Test ROI computation with scale factor applied."""
        roinodes = [
            (0.0, 0.0, 0.0, 10.0),
        ]
        result = roi_compute(roinodes, scalfac=2.0, mode="average")
        assert abs(result - 20.0) < 0.001

    def test_roi_unsupported_mode(self):
        """Test unsupported mode returns None."""
        roinodes = [(0.0, 0.0, 0.0, 1.0)]
        result = roi_compute(roinodes, mode="unsupported_mode")
        assert result is None


class TestNodeValueFromImage:
    """Tests for calculate_node_value_from_image."""

    def test_average_stat_type(self):
        """Test average stat type on image data."""
        image = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        mask = np.ones((3, 3), dtype=bool)
        result = calculate_node_value_from_image(image, mask, stat_type="average")
        assert abs(result - 5.0) < 0.001

    def test_max_stat_type(self):
        """Test max stat type on image data."""
        image = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        mask = np.ones((2, 2), dtype=bool)
        result = calculate_node_value_from_image(image, mask, stat_type="max")
        assert abs(result - 4.0) < 0.001

    def test_min_stat_type(self):
        """Test min stat type on image data."""
        image = np.array([
            [5.0, 3.0],
            [7.0, 2.0],
        ])
        mask = np.ones((2, 2), dtype=bool)
        result = calculate_node_value_from_image(image, mask, stat_type="min")
        assert abs(result - 2.0) < 0.001

    def test_filteraverage_stat_type(self):
        """Test filteraverage stat type (mean within ±1 std)."""
        image = np.array([1.0, 2.0, 3.0, 4.0, 100.0]).reshape(5, 1)
        mask = np.ones((5, 1), dtype=bool)
        result = calculate_node_value_from_image(image, mask, stat_type="filteraverage")
        assert 0 < result < 100, (
            "Filteraverage should exclude outlier"
        )

    def test_partial_mask(self):
        """Test with partial mask selecting only some voxels."""
        image = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])
        mask = np.array([
            [True, False],
            [False, True],
        ])
        result = calculate_node_value_from_image(image, mask, stat_type="average")
        assert abs(result - 25.0) < 0.001, (
            f"Expected 25.0 (mean of 10 and 40), got {result}"
        )

    def test_unknown_stat_type(self):
        """Test unknown stat type returns None."""
        image = np.zeros((3, 3))
        mask = np.ones((3, 3), dtype=bool)
        result = calculate_node_value_from_image(image, mask, stat_type="unknown")
        assert result is None


class TestNode2Connect:
    """Tests for node2connect function."""

    def test_node2connect_values_sum_to_one(self):
        """Test computed connectivity values sum to approximately 1."""
        nodelist = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "value": 0.0, "size": 1.0, "label": "A"},
            {"x": 1.0, "y": 0.0, "z": 0.0, "value": 0.0, "size": 1.0, "label": "B"},
            {"x": 0.0, "y": 1.0, "z": 0.0, "value": 0.0, "size": 1.0, "label": "C"},
        ]
        matrix = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        result = node2connect(nodelist, matrix)
        values = [node["value"] for node in result]
        assert abs(sum(values) - 1.0) < 0.001, (
            f"Values should sum to 1, got {sum(values)}"
        )

    def test_node2connect_preserves_other_fields(self):
        """Test node2connect preserves x, y, z, size, label fields."""
        nodelist = [
            {"x": 1.0, "y": 2.0, "z": 3.0, "value": 0.0, "size": 4.0, "label": "Test"},
        ]
        matrix = np.array([[1.0]])
        result = node2connect(nodelist, matrix)
        assert result[0]["x"] == 1.0
        assert result[0]["y"] == 2.0
        assert result[0]["z"] == 3.0
        assert result[0]["size"] == 4.0
        assert result[0]["label"] == "Test"
