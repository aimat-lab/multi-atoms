"""PolyAtoms - Multi-worker abstraction layer on top of MultiAtoms."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, TypeVar

from multiatoms.core import (
    BatchedAtoms,
    MultiAtoms,
)
from multiatoms.model_manager import (
    PolyAtomModelManagerDecorator,
)

U = TypeVar("U")


def _chunk_list(flat_list: List, chunk_size: int) -> List[List]:
    """Split a flat list into chunks of the given size."""
    return [flat_list[i : i + chunk_size] for i in range(0, len(flat_list), chunk_size)]


class PolyAtoms:
    """Manages multiple MultiAtoms instances across workers.

    Provides the same API as MultiAtoms but distributes work across n_workers,
    each with their own MultiAtoms instance. All methods return flat lists
    to maintain API compatibility with MultiAtoms.

    Args:
        n_workers: Number of MultiAtoms workers to create
        **kwargs: Arguments forwarded to each MultiAtoms instance
    """

    def __init__(self, n_workers: int = 1, **kwargs):
        self._n_workers = n_workers
        self._n_systems_per_worker = kwargs.get("n_systems", 1)
        kwargs["model_manager"] = PolyAtomModelManagerDecorator(kwargs["model_manager"])

        self._workers: List[MultiAtoms] = [
            MultiAtoms(**kwargs) for _ in range(n_workers)
        ]

    def __getattr__(self, name: str) -> "PolyAtomAttribute":
        # Check if attribute exists on MultiAtoms
        if not hasattr(self._workers[0], name):
            raise AttributeError(f"No attribute '{name}'")

        return PolyAtomAttribute(self._workers, name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            for worker in self._workers:
                setattr(worker, name, value)

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def n_systems(self) -> int:
        """Total number of systems across all workers."""
        return self._n_workers * self._n_systems_per_worker

    @property
    def workers(self) -> List[MultiAtoms]:
        return self._workers

    @property
    def atoms(self) -> List[BatchedAtoms]:
        """Flat list of all BatchedAtoms across all workers."""
        result = []
        for worker in self._workers:
            result.extend(worker.atoms)
        return result

    @contextmanager
    def parallel(self):
        """Context manager to enable parallel execution on all workers."""
        for worker in self._workers:
            worker._parallel_mode = True
            for atoms in worker._atoms_list:
                atoms._parallel_mode = True
        try:
            yield
        finally:
            for worker in self._workers:
                worker._parallel_mode = False
                for atoms in worker._atoms_list:
                    atoms._parallel_mode = False

    def map(self, func: Callable[..., U], *iterables: Iterable) -> List[U]:
        """Apply function across all workers' atoms, returning a flat list.

        Iterables are automatically chunked and distributed to workers based on
        n_systems_per_worker. Results from all workers are flattened into a
        single list.
        """
        iterables_list = [list(it) for it in iterables]
        chunked = [_chunk_list(it, self._n_systems_per_worker) for it in iterables_list]

        results = []
        for worker_idx, worker in enumerate(self._workers):
            worker_iterables = [c[worker_idx] for c in chunked]
            results.extend(worker.map(func, *worker_iterables))
        return results

    def foreach(self, func: Callable[..., Any], *iterables: Iterable) -> None:
        """Apply function across all workers' atoms.

        Iterables are automatically chunked and distributed to workers based on
        n_systems_per_worker.
        """
        iterables_list = [list(it) for it in iterables]
        chunked = [_chunk_list(it, self._n_systems_per_worker) for it in iterables_list]

        for worker_idx, worker in enumerate(self._workers):
            worker_iterables = [c[worker_idx] for c in chunked]
            worker.foreach(func, *worker_iterables)

    def clean_up(self):
        """Clean up all worker resources."""
        for worker in self._workers:
            worker.clean_up()


class PolyAtomAttribute:
    """Proxy for accessing attributes across all workers' MultiAtoms."""

    def __init__(self, workers: List[MultiAtoms], attr_name: str):
        self._workers = workers
        self._attr_name = attr_name

    def __getitem__(self, key):
        """Delegate indexing to all workers, flatten results."""
        results = []
        for worker in self._workers:
            worker_attr = getattr(worker, self._attr_name)
            if hasattr(worker_attr, "__getitem__"):
                results.extend(worker_attr[key])
            else:
                results.append(worker_attr[key])
        return results

    def __setitem__(self, key, value):
        """Delegate item assignment to all workers."""
        for worker in self._workers:
            getattr(worker, self._attr_name)[key] = value

    def __call__(self, *args, **kwargs):
        """Delegate method calls to all workers, flatten results."""
        results = []
        for worker in self._workers:
            result = getattr(worker, self._attr_name)(*args, **kwargs)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results
