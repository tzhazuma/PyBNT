"""Shared test fixtures for PyBrainViewer unit tests."""
import pytest
import os
from pathlib import Path

import pybnt

TEMPLATES_DIR = Path(pybnt.__file__).parent / "data" / "templates"


@pytest.fixture
def icbm152_surface():
    """Path to ICBM152 surface template file (.nv format)."""
    path = TEMPLATES_DIR / "BrainMesh_ICBM152.nv"
    if not path.exists():
        pytest.skip(f"ICBM152 surface file not found: {path}")
    return str(path)


@pytest.fixture
def icbm152_smoothed_surface():
    """Path to ICBM152 smoothed surface template."""
    path = TEMPLATES_DIR / "BrainMesh_ICBM152_smoothed.nv"
    if not path.exists():
        pytest.skip(f"ICBM152 smoothed surface file not found: {path}")
    return str(path)


@pytest.fixture
def ch2_surface():
    """Path to Ch2 surface template."""
    path = TEMPLATES_DIR / "BrainMesh_Ch2.nv"
    if not path.exists():
        pytest.skip(f"Ch2 surface file not found: {path}")
    return str(path)


@pytest.fixture
def aal90_node():
    """Path to AAL90 node file."""
    path = TEMPLATES_DIR / "AAL90" / "Node_AAL90.node"
    if not path.exists():
        pytest.skip(f"AAL90 node file not found: {path}")
    return str(path)


@pytest.fixture
def aal90_edge_binary():
    """Path to AAL90 binary edge file."""
    path = TEMPLATES_DIR / "AAL90" / "Edge_AAL90_Binary.edge"
    if not path.exists():
        pytest.skip(f"AAL90 binary edge file not found: {path}")
    return str(path)


@pytest.fixture
def aal90_edge_weighted():
    """Path to AAL90 weighted edge file."""
    path = TEMPLATES_DIR / "AAL90" / "Edge_AAL90_Weighted.edge"
    if not path.exists():
        pytest.skip(f"AAL90 weighted edge file not found: {path}")
    return str(path)


@pytest.fixture
def power264_node():
    """Path to Power264 node file."""
    path = TEMPLATES_DIR / "Power264" / "Node_Power264.node"
    if not path.exists():
        pytest.skip(f"Power264 node file not found: {path}")
    return str(path)


@pytest.fixture
def templates_dir():
    """Path to the templates directory."""
    return str(TEMPLATES_DIR)
