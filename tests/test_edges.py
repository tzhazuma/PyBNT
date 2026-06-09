"""Tests for pybnt/core/edges.py — edge file parsing and construction."""

import pytest
import numpy as np
from pybnt.core.edges import parse_edge_file, make_edge_from_nodes


class TestParseEdgeFile:
    """Tests for parse_edge_file function."""

    def test_parse_binary_edge_file(self, aal90_edge_binary):
        """Test parsing AAL90 binary edge file returns a 2D array."""
        matrix = parse_edge_file(aal90_edge_binary)
        assert matrix is not None, "Should return a matrix"
        assert isinstance(matrix, np.ndarray), "Should be numpy array"
        assert len(matrix.shape) == 2, "Edge matrix should be 2D"

    def test_edge_matrix_is_square(self, aal90_edge_binary):
        """Test edge matrix is square (N x N)."""
        matrix = parse_edge_file(aal90_edge_binary)
        assert matrix is not None
        assert matrix.shape[0] == matrix.shape[1], (
            f"Edge matrix should be square, got {matrix.shape}"
        )

    def test_edge_matrix_aal90_size(self, aal90_edge_binary):
        """Test AAL90 edge matrix is 90x90."""
        matrix = parse_edge_file(aal90_edge_binary)
        assert matrix is not None
        assert matrix.shape == (90, 90), (
            f"AAL90 edge matrix should be 90x90, got {matrix.shape}"
        )

    def test_binary_edge_contains_ints(self, aal90_edge_binary):
        """Test binary edge matrix values are numeric."""
        matrix = parse_edge_file(aal90_edge_binary)
        assert matrix is not None
        assert np.issubdtype(matrix.dtype, np.number), (
            "Edge values should be numeric"
        )

    def test_parse_weighted_edge_file(self, aal90_edge_weighted):
        """Test parsing weighted edge file."""
        matrix = parse_edge_file(aal90_edge_weighted)
        assert matrix is not None
        assert isinstance(matrix, np.ndarray)
        assert len(matrix.shape) == 2

    def test_continue_mode_skips_malformed(self, tmp_path):
        """Test 'continue' mode skips malformed lines."""
        path = tmp_path / "mixed.edge"
        path.write_text("1\t2\t3\ninvalid\tline\na\n4\t5\t6\n")
        matrix = parse_edge_file(str(path), mode="continue")
        assert matrix is not None
        assert len(matrix) >= 1, "Should have at least one valid row"

    def test_strict_mode_returns_none_on_error(self, tmp_path):
        """Test strict mode returns None on first malformed line."""
        path = tmp_path / "bad.edge"
        path.write_text("1\t2\t3\nbad\tdata\n")
        matrix = parse_edge_file(str(path), mode="strict")
        assert matrix is None, "Strict mode should return None on error"

    def test_parse_single_line(self, tmp_path):
        """Test parsing a single-line edge file."""
        path = tmp_path / "single.edge"
        path.write_text("0.5\t1.0\t0.0\n")
        matrix = parse_edge_file(str(path))
        assert matrix is not None
        assert matrix.shape == (1, 3), f"Expected (1, 3), got {matrix.shape}"
        assert matrix[0, 0] == 0.5
        assert matrix[0, 1] == 1.0
        assert matrix[0, 2] == 0.0

    def test_parse_tab_separated_values(self, tmp_path):
        """Test parsing tab-separated edge values."""
        path = tmp_path / "tabs.edge"
        path.write_text("1\t2\t3\n")
        matrix = parse_edge_file(str(path))
        assert matrix is not None
        assert len(matrix) == 1, f"Expected 1 row, got {len(matrix)}"
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 2.0
        assert matrix[0, 2] == 3.0


class TestMakeEdgeFromNodes:
    """Tests for make_edge_from_nodes function."""

    def test_make_edge_binary_pairs(self, tmp_path):
        """Test creating edge matrix from binary (i, j) pairs."""
        out_path = str(tmp_path / "test_binary.edge")
        result = make_edge_from_nodes(
            size=(3, 3),
            outpath=out_path,
            pairnodesl=[(0, 1), (1, 2), (2, 0)],
        )
        assert result is not None
        assert result.shape == (3, 3)
        assert result[0, 1] == 1.0, "Edge (0,1) should be 1"
        assert result[0, 0] == 0.0, "No self-loop at (0,0)"
        assert result[1, 2] == 1.0, "Edge (1,2) should be 1"
        assert result[2, 0] == 1.0, "Edge (2,0) should be 1"

    def test_make_edge_weighted_pairs(self, tmp_path):
        """Test creating edge matrix from weighted (i, j, w) triples."""
        out_path = str(tmp_path / "test_weighted.edge")
        result = make_edge_from_nodes(
            size=(2, 2),
            outpath=out_path,
            pairnodesl=[(0, 1, 0.5), (1, 0, 0.8)],
        )
        assert result is not None
        assert result[0, 1] == 0.5, "Weighted edge should be 0.5"
        assert result[1, 0] == 0.8, "Weighted edge should be 0.8"

    def test_make_edge_mixed_pairs(self, tmp_path):
        """Test creating edge matrix from mixed binary and weighted pairs."""
        out_path = str(tmp_path / "test_mixed.edge")
        result = make_edge_from_nodes(
            size=(2, 2),
            outpath=out_path,
            pairnodesl=[(0, 1), (1, 0, 3.14)],
        )
        assert result is not None
        assert result[0, 1] == 1.0, "Binary edge should be 1"
        assert result[1, 0] == 3.14, "Weighted edge should be 3.14"

    def test_make_edge_output_file_exists(self, tmp_path):
        """Test output file is created."""
        out_path = str(tmp_path / "test_output.edge")
        make_edge_from_nodes(
            size=(1, 1),
            outpath=out_path,
            pairnodesl=[],
        )
        import os
        assert os.path.exists(out_path), "Output file should be created"

    def test_make_edge_output_file_content(self, tmp_path):
        """Test output file contains correct space-separated values."""
        out_path = str(tmp_path / "test_content.edge")
        make_edge_from_nodes(
            size=(2, 2),
            outpath=out_path,
            pairnodesl=[(0, 1, 1.5)],
        )
        with open(out_path) as f:
            content = f.read()
        assert "0.0" in content, "Unset edges should be 0.0"
        assert "1.5" in content, "Weighted edge should appear in file"

    def test_make_edge_empty_pairs(self, tmp_path):
        """Test creating edge matrix with empty pair list."""
        out_path = str(tmp_path / "test_empty.edge")
        result = make_edge_from_nodes(
            size=(3, 3),
            outpath=out_path,
            pairnodesl=[],
        )
        assert result is not None
        assert np.all(result == 0.0), "Empty pairs should give all-zero matrix"

    def test_make_edge_self_loop(self, tmp_path):
        """Test creating self-loop edge."""
        out_path = str(tmp_path / "test_self.edge")
        result = make_edge_from_nodes(
            size=(2, 2),
            outpath=out_path,
            pairnodesl=[(0, 0, 1.0)],
        )
        assert result is not None
        assert result[0, 0] == 1.0, "Self-loop should be set"
        assert result[1, 1] == 0.0, "Non-specified diagonal stays 0"

    def test_make_edge_pair_length_validation(self, tmp_path):
        """Test that make_edge_from_nodes handles pair length > 3."""
        out_path = str(tmp_path / "test_length.edge")
        result = make_edge_from_nodes(
            size=(1, 1),
            outpath=out_path,
            pairnodesl=[(0, 0, 0.5, 0.9)],  # length 4 tuple
        )
        assert result is not None
