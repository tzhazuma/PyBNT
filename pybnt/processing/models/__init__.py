"""Super-resolution model architectures (EDSR, Combi)."""
from pybnt.processing.models.edsr import edsr, combi
from pybnt.processing.models.common import resolve, resolve_single, evaluate
from pybnt.processing.models.train import EdsrTrainer
