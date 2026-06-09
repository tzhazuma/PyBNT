"""PyBrainViewer - AI-Powered Multi-Function Brain Visual Tool"""
__version__ = "0.2.0"
__author__ = "Tang Zhihao"

# Lazy imports so visualization can be used without full dependency chain
_submodules = {
    "core": "pybnt.core",
    "visualization": "pybnt.visualization",
    "processing": "pybnt.processing",
    "ai": "pybnt.ai",
    "external": "pybnt.external",
    "gui": "pybnt.gui",
    "cli": "pybnt.cli",
    "tui": "pybnt.tui",
}


def __getattr__(name):
    if name in _submodules:
        import importlib
        module = importlib.import_module(_submodules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
