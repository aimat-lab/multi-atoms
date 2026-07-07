"""Core classes for managing multiple parallel molecular dynamics simulations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, TypeVar

import ase.io
import numpy as np
from ase import Atoms
from greenlet import greenlet

from multiatoms.proxy_calculator import (
    ProxyCalculator,
)
from multiatoms.scheduler import HubScheduler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiatoms.model_manager import (
        ModelManager,
    )


U = TypeVar("U")


def _validate_iterables(iterables: tuple[Iterable, ...]) -> List[List]:
    """Convert iterables to lists and validate they have the same length.

    Args:
        iterables: Tuple of iterables to validate

    Returns:
        List of lists from the iterables

    Raises:
        ValueError: If no iterables provided or lengths don't match
    """
    iterables_list = [list(it) for it in iterables]
    if not iterables_list:
        raise ValueError("At least one iterable must be provided")

    lengths = [len(it) for it in iterables_list]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All iterables must have the same length. Got lengths: {lengths}"
        )

    return iterables_list


class BatchedAtoms(Atoms):
    """Atoms subclass that hooks get_forces() to yield or call GPU directly."""

    def __init__(
        self,
        model_manager: ModelManager,
        scheduler: HubScheduler,
        template: Atoms | None = None,
        **kwargs,
    ):
        """Initialize BatchedAtoms.

        Args:
            model_manager: ModelManager for GPU computation
            scheduler: HubScheduler for parallel mode coordination
            template: Optional ASE ``Atoms`` to copy full state from -- positions,
                cell, periodic boundary conditions, constraints, charges. The copy
                is independent, so systems never share arrays. When given,
                ``kwargs`` are ignored.
            **kwargs: Arguments passed to the ASE Atoms constructor
                (symbols, positions, ...) when no template is given.
        """
        if template is not None:
            super().__init__(template)
        else:
            super().__init__(**kwargs)
        self.model_manager = model_manager
        self.scheduler = scheduler
        self._parallel_mode: bool = False
        self._cached_positions: np.ndarray | None = None

    def get_forces(self, apply_constraint=True, md=False):
        """Get forces, either via batched forward or direct GPU call.

        In parallel mode, yields to scheduler which batches GPU calls.
        In single mode, directly calls ModelManager.
        """
        if self._parallel_mode:
            # Yield control to scheduler, passing self
            self.scheduler.yield_to_next(self)
            # When we resume, ProxyCalculator has our results
        else:
            # Direct call for single-atom mode
            self.model_manager.compute_energy_and_forces([self])

        return super().get_forces(apply_constraint, md)


class MultiAtoms:
    """Manages multiple parallel molecular dynamics simulations."""

    def __init__(
        self,
        template: Atoms | str | Path,
        model_manager: ModelManager,
        n_systems: int = 1,
    ):
        """Initialize MultiAtoms with parallel simulation systems.

        Creates BatchedAtoms instances from a template structure and sets up
        the hub scheduler and model manager for batched computation.

        Args:
            template: The system to replicate, as an ASE ``Atoms`` object or a
                path to any ASE-readable structure file (PDB, xyz, CIF, ...).
                Its full state -- cell, periodic boundary conditions,
                constraints, charges -- is copied into every system.
            model_manager: ModelManager for model input/output handling
            n_systems: Number of parallel systems to create
        """
        if isinstance(template, Atoms):
            template_atoms = template.copy()
        else:
            template_atoms = ase.io.read(template)
        n_atoms = len(template_atoms)

        logger.info(
            f"Creating MultiAtoms with {n_systems} systems "
            f"on device {model_manager.device}"
        )

        # Create GPU forward manager and scheduler
        self._model_manager = model_manager
        self._scheduler = HubScheduler(self._model_manager)
        self._batch_processor = model_manager
        self._n_systems = n_systems

        # Each BatchedAtoms deep-copies the template, so cell/pbc/constraints
        # are preserved and the systems stay fully independent.
        self._atoms_list: List[BatchedAtoms] = []

        for i in range(n_systems):
            batched_atom = BatchedAtoms(
                model_manager=self._model_manager,
                scheduler=self._scheduler,
                template=template_atoms,
            )
            batched_atom.calc = ProxyCalculator(n_atoms=n_atoms)
            self._atoms_list.append(batched_atom)

        self._parallel_mode: bool = False

    def __getattr__(self, name: str):
        if not hasattr(self._atoms_list[0], name):
            raise AttributeError(f"No attribute '{name}'")

        attr = getattr(self._atoms_list[0], name)

        if callable(attr) or isinstance(attr, (dict, list)):
            return MultiAtomAttribute(self._atoms_list, name)

        # Only return raw values if it's a simple property (like .positions, .cell)
        return [getattr(atom, name) for atom in self._atoms_list]

    def __setattr__(self, name: str, value: Any):
        """Intercept attribute assignment - set on all atoms."""
        if name.startswith("_"):
            # These belong to MultiAtoms itself
            object.__setattr__(self, name, value)
        else:
            # Delegate to all atoms
            for atoms in self._atoms_list:
                setattr(atoms, name, value)

    @property
    def n_systems(self) -> int:
        """Returns number of systems in multi atoms object."""
        return self._n_systems

    @property
    def atoms(self) -> List[BatchedAtoms]:
        """Returns the list of atoms objects."""
        return self._atoms_list

    @contextmanager
    def parallel(self):
        """Context manager to enable parallel execution of map/foreach operations."""
        self._parallel_mode = True
        for atoms in self._atoms_list:
            atoms._parallel_mode = True
        try:
            yield
        finally:
            self._parallel_mode = False
            for atoms in self._atoms_list:
                atoms._parallel_mode = False

    def map(self, func: Callable[..., U], *iterables: Iterable) -> List[U]:
        """Apply function to elements from one or more iterables and return results.

        Args:
            func: Function to apply to each set of elements
            *iterables: One or more iterables to zip together

        Returns:
            List of results from applying func to each zipped tuple

        Example:
            multi_atoms.map(lambda a: create_monitor(a), multi_atoms.atoms)
            multi_atoms.map(lambda a, m: (a, m), multi_atoms.atoms, monitors)
        """
        iterables_list = _validate_iterables(iterables)

        if self._parallel_mode:
            return self._parallel_map(func, iterables_list)
        else:
            return [func(*args) for args in zip(*iterables_list)]

    def foreach(self, func: Callable[..., Any], *iterables: Iterable) -> None:
        """Apply function to elements from one or more iterables (no return values).

        Args:
            func: Function to apply to each set of elements
            *iterables: One or more iterables to zip together

        Example:
            multi_atoms.foreach(lambda a: print(a), multi_atoms.atoms)
            multi_atoms.foreach(lambda a, t: t.attach(a), multi_atoms.atoms, trajs)
        """
        iterables_list = _validate_iterables(iterables)

        if self._parallel_mode:
            self._parallel_foreach(func, iterables_list)
        else:
            for args in zip(*iterables_list):
                func(*args)

    def _run_greenlets(self, greenlets: List[greenlet]) -> None:
        """Run a pool of greenlets through the scheduler."""
        self._scheduler.set_greenlet_pool(greenlets)
        self._scheduler.kick_off()
        self._scheduler.clear()

    def _parallel_map(
        self, func: Callable[..., Any], iterables_list: List[List]
    ) -> List[Any]:
        """Execute function on zipped iterables in parallel using greenlets."""
        n_items = len(iterables_list[0])
        results = [None] * n_items

        def worker(idx, *args):
            results[idx] = func(*args)

        greenlets = [
            greenlet(partial(worker, idx, *args))
            for idx, args in enumerate(zip(*iterables_list))
        ]
        self._run_greenlets(greenlets)
        return results

    def _parallel_foreach(
        self, func: Callable[..., Any], iterables_list: List[List]
    ) -> None:
        """Execute function on zipped iterables in parallel using greenlets."""
        greenlets = [greenlet(partial(func, *args)) for args in zip(*iterables_list)]
        self._run_greenlets(greenlets)

    def clean_up(self):
        """Clean up model resources."""
        self._batch_processor.cleanup()


class MultiAtomAttribute:
    """Proxy for accessing attributes on multiple atoms."""

    def __init__(self, atoms_list: List[BatchedAtoms], attr_name: str):
        self.atoms_list = atoms_list
        self.attr_name = attr_name

    def __getitem__(self, key):
        """Allow: multi_atoms.arrays['partial_charges'] = value"""
        values = []
        for atoms in self.atoms_list:
            values.append(getattr(atoms, self.attr_name)[key])
        return values

    def __call__(self, *args, **kwargs):
        """Allow: multi_atoms.get_positions()"""
        return [
            getattr(atoms, self.attr_name)(*args, **kwargs) for atoms in self.atoms_list
        ]

    def __setitem__(self, key, value):
        for atoms in self.atoms_list:
            getattr(atoms, self.attr_name)[key] = value
