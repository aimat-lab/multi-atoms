"""MultiAtoms module for parallel molecular dynamics simulations.

This module provides tools for running multiple MD simulations in parallel,
batching GPU computations across systems for improved throughput.

Main classes:
    - MultiAtoms: High-level API for managing parallel simulations
    - PolyAtoms: Run K parallel MultiAtoms in separate processes, sharing one GPU
    - BatchedAtoms: ASE Atoms subclass that supports batched GPU calls
    - ModelManager: Abstract base class for batched model input/output processing
    - HubScheduler: Greenlet-based cooperative scheduler
"""

from multiatoms.core import (
    BatchedAtoms,
    MultiAtomAttribute,
    MultiAtoms,
)
from multiatoms.model_manager import (
    ModelManager,
)
from multiatoms.poly_atoms import (
    PolyAtoms,
    RemoteModelManager,
)
from multiatoms.scheduler import HubScheduler

__all__ = [
    "BatchedAtoms",
    "ModelManager",
    "MultiAtomAttribute",
    "MultiAtoms",
    "PolyAtoms",
    "RemoteModelManager",
    "HubScheduler",
]
