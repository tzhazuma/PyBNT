"""Integration tests using real PyBNT template data.

These tests exercise the full pipeline — parsing real surface, node, and
edge files from the bundled template data (ICBM152, AAL90), projecting
nodes onto surfaces, and verifying that the CLI, AI modules, and project
configuration are all consistent and importable.

By default none of these tests are marked ``slow`` since they all complete
in well under one minute on modern hardware.
"""

import pytest
import numpy as np
from pathlib import Path

import pybnt

TEMPLATES = Path(pybnt.__file__).parent / "data" / "templates"
ICBM152 = TEMPLATES / "BrainMesh_ICBM152.nv"
AAL90_NODE = TEMPLATES / "AAL90" / "Node_AAL90.node"
AAL90_EDGE = TEMPLATES / "AAL90" / "Edge_AAL90_Binary.edge"


# ── helpers ────────────────────────────────────────────────────────────


def skip_if_missing(*paths: Path) -> None:
    """Pytest-skip if any of the given template paths do not exist."""
    for p in paths:
        if not p.exists():
            pytest.skip(f"Template file not found: {p}")


# ── Surface pipeline ───────────────────────────────────────────────────


class TestSurfacePipeline:
    """End-to-end surface parsing and visualisation mesh creation."""

    def test_icbm152_parse(self):
        """Integration: parse ICBM152 surface with correct vertex + triangle counts."""
        skip_if_missing(ICBM152)
        from pybnt.core.surface import parse_surface

        nodes, tris = parse_surface(str(ICBM152))
        assert len(nodes) == 81924
        assert len(tris) == 163840
        assert nodes.shape[1] == 3

    def test_surface_to_visualization(self):
        """Integration: parse surface and create a pyvista mesh."""
        skip_if_missing(ICBM152)
        from pybnt.core.surface import parse_surface
        import pyvista as pv

        nodes, tris = parse_surface(str(ICBM152))
        mesh = pv.make_tri_mesh(np.array(nodes), np.array(tris))
        assert mesh.n_points == 81924
        assert mesh.n_cells == 163840


# ── Network pipeline ───────────────────────────────────────────────────


class TestNetworkPipeline:
    """End-to-end node parsing, edge parsing, and surface projection."""

    def test_aal90_node_parse_and_project(self):
        """Integration: parse AAL90 nodes and project to ICBM152 surface."""
        skip_if_missing(AAL90_NODE, ICBM152)
        from pybnt.core.nodes import parse_node_file
        from pybnt.core.surface import parse_surface
        from pybnt.core.projection import Triangle, Surfaces

        nodes = parse_node_file(str(AAL90_NODE))
        assert len(nodes) == 90

        surface_nodes, tris = parse_surface(str(ICBM152))
        # BNV .nv format uses 1-based triangle indices — convert to 0-based.
        # Use a subset of triangles for the projection step to keep the test
        # fast (the full 163k-triangle projection is prohibitively slow).
        tris_zero = np.array(tris) - 1
        sample = tris_zero[:500]
        tl = []
        for tri_idx in sample:
            vertices = tuple(surface_nodes[int(idx)] for idx in tri_idx)
            tl.append(Triangle(vertices, nodesindex=tuple(int(i) for i in tri_idx)))
        sf = Surfaces(tl)
        sf.proj_surface_all(ps=nodes, proj_type="nearest")
        assert len(sf.tri) == 500

    def test_edge_matrix_compatible_with_nodes(self):
        """Integration: edge matrix dimension matches node count."""
        skip_if_missing(AAL90_EDGE, AAL90_NODE)
        from pybnt.core.nodes import parse_node_file
        from pybnt.core.edges import parse_edge_file

        nodes = parse_node_file(str(AAL90_NODE))
        edges = parse_edge_file(str(AAL90_EDGE))
        assert edges.shape[0] == edges.shape[1], "Edge matrix must be square"
        assert edges.shape[0] == len(nodes), (
            f"Edge dims ({edges.shape[0]}) must match node count ({len(nodes)})"
        )


# ── Volume mapping ─────────────────────────────────────────────────────


class TestVolumeMapping:
    """Volume-to-surface mapping algorithm completeness."""

    def test_all_volume_algorithms_available(self):
        """Integration: all documented volume mapping algorithms are recognised."""
        from pybnt.core.volume import map_volume_to_surface

        expected = {
            "nearest",
            "linear",
            "gaussian",
            "cubic",
            "maximum",
            "minimum",
            "average",
        }
        # The algorithms dict is internal; we verify they're accepted by
        # inspecting the function's docstring or the known set above.
        import inspect

        doc = inspect.getdoc(map_volume_to_surface) or ""
        # Check that all expected algorithm names are documented
        for alg in sorted(expected):
            assert alg in doc, (
                f"Algorithm {alg!r} not found in map_volume_to_surface docstring"
            )


# ── AI module imports ──────────────────────────────────────────────────


class TestAIModules:
    """Verify AI sub-modules import cleanly (no missing dependencies)."""

    def test_llm_module_imports(self):
        """AI: LLM module imports cleanly."""
        from pybnt.ai.llm import ask_llm, ask_llm_openode, ask_llm_multimodal

        assert callable(ask_llm)
        assert callable(ask_llm_multimodal)

    def test_rag_module_imports(self):
        """AI: RAG module imports cleanly."""
        from pybnt.ai.rag import BrainKnowledgeStore, query_knowledge

        assert callable(query_knowledge)

    def test_vlm_module_imports(self):
        """AI: VLM module imports cleanly."""
        from pybnt.ai.vlm import BrainMRISigLIP, describe_brain_image

        assert callable(describe_brain_image)


# ── CLI command structure ──────────────────────────────────────────────


class TestCLICommands:
    """CLI command-group structure via Click."""

    def test_cli_ai_commands_exist(self):
        """CLI: ``ai`` command group has all documented subcommands."""
        from pybnt.cli.main import cli

        ai_cmd = cli.commands.get("ai")
        assert ai_cmd is not None, "ai command group not found"
        cmd_names = list(ai_cmd.commands.keys())
        for cmd in ["segment", "correct", "ask"]:
            assert cmd in cmd_names, f"Missing CLI command: ai {cmd}"

    def test_cli_visualize_commands_exist(self):
        """CLI: ``visualize`` command group has expected subcommands."""
        from pybnt.cli.main import cli

        viz_cmd = cli.commands.get("visualize")
        assert viz_cmd is not None, "visualize command group not found"
        cmd_names = list(viz_cmd.commands.keys())
        for cmd in ["surface", "nodes"]:
            assert cmd in cmd_names, f"Missing CLI command: visualize {cmd}"

    def test_cli_process_commands_exist(self):
        """CLI: ``process`` command group has expected subcommands."""
        from pybnt.cli.main import cli

        proc_cmd = cli.commands.get("process")
        assert proc_cmd is not None, "process command group not found"
        cmd_names = list(proc_cmd.commands.keys())
        for cmd in ["convert", "compute-connectivity", "register", "superres"]:
            assert cmd in cmd_names, f"Missing CLI command: process {cmd}"

    def test_cli_volume_command_exists(self):
        """CLI: top-level ``volume`` command is registered."""
        from pybnt.cli.main import cli

        assert "volume" in cli.commands, "Missing top-level CLI command: volume"


# ── Project configuration ──────────────────────────────────────────────


class TestProjectConfig:
    """Project-level configuration consistency checks."""

    def test_pyproject_version_consistency(self):
        """Config: pyproject.toml version matches package ``__version__``."""
        import tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        assert config["project"]["version"] == pybnt.__version__, (
            f"pyproject.toml version ({config['project']['version']}) "
            f"does not match pybnt.__version__ ({pybnt.__version__})"
        )

    def test_entry_points_configured(self):
        """Config: CLI and GUI entry points are configured."""
        import tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        scripts = config["project"]["scripts"]
        assert "pybnt" in scripts, "Missing pybnt CLI entry point"
        assert "pybnt-gui" in scripts, "Missing pybnt-gui entry point"
