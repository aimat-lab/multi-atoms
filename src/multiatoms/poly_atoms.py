"""Multi-process GPU force server: K parallel ``MultiAtoms`` sharing one GPU.

A single ``MultiAtoms`` run alternates synchronously between the GPU (one batched
forward for all systems) and the CPU (every ASE integrator does its step). While
the integrators step, the GPU is idle -- measured at ~45% of wall-clock for a
SchNet potential. ``PolyAtoms`` reclaims that idle time: it runs ``workers``
independent ``MultiAtoms`` simulations in separate processes, each shipping its
force requests to one central GPU server. While worker A integrates on the CPU,
the GPU is busy with worker B's batch. Benchmarked at ~1.8x throughput on one
A100 with K=2.

Design
------
* The **main process is the server** -- it holds the live ``ModelManager`` on the
  GPU (built normally, never pickled across a process boundary) and answers force
  requests one batch at a time.
* Each **worker** is an ordinary ``MultiAtoms`` driven by the unchanged greenlet
  scheduler / ``BatchedAtoms`` / ``ProxyCalculator``. Its only difference is a
  ``RemoteModelManager`` that ships positions to the server instead of running a
  model locally. Workers are CPU-only -> exactly one CUDA context lives in main.
* ``spawn`` is used (fork + CUDA is unsafe); spawned workers start clean and do
  not inherit main's CUDA context.

Usage
-----
The per-worker simulation is a **top-level function** ``fn(multi, worker_id)``
(so standard pickle can ship it; bind config with ``functools.partial``). Inside
it, ``multi`` is a real local ``MultiAtoms`` with the full API -- forces are
transparently batched on the shared GPU::

    def simulate(multi, worker_id):
        multi.foreach(lambda a: MaxwellBoltzmannDistribution(a, temperature_K=300),
                      multi.atoms)
        integrators = multi.map(lambda a: SmartLangevin(a, ...), multi.atoms)
        with multi.parallel():
            multi.foreach(lambda i: i.run(1000), integrators)
        return multi.get_positions()

    if __name__ == "__main__":                  # required for spawn
        with PolyAtoms(pdb_path, manager, n_systems=256, workers=2) as poly:
            results = poly.run(simulate, seeds=[0, 1])   # list of 2 results

NOTE: build the GPU model/manager under ``if __name__ == "__main__"`` so the
worker re-import of your script does not rebuild it on the GPU.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Callable, List, Optional

import ase.io
import numpy as np

from multiatoms.core import MultiAtoms
from multiatoms.model_manager import ModelManager

# Message kinds on the shared request queue (worker -> server).
_KIND_FORCES = 0  # payload: positions array (n_systems, n_atoms, 3)
_KIND_DONE = 1  # payload: the worker's fn result (or _WorkerError)

# How long the server waits for a message before checking worker liveness. Only
# affects how quickly a hard worker death is noticed, not steady-state
# throughput -- a pending message is returned immediately.
_POLL_TIMEOUT_S = 5.0


class _WorkerError:
    """Carries a worker-side traceback back to the main process."""

    def __init__(self, message: str):
        self.message = message


class _ServerError:
    """Carries a server-side (model) traceback back to a worker."""

    def __init__(self, message: str):
        self.message = message


class RemoteModelManager(ModelManager):
    """ModelManager that ships force requests to the GPU server in the main process.

    Overrides only ``_infer`` -- the cache filter and result distribution in
    ``compute_energy_and_forces`` are inherited unchanged, as are the scheduler,
    ``BatchedAtoms`` and ``ProxyCalculator`` that drive it.
    """

    def __init__(self, req_q, res_q, worker_id: int):
        super().__init__(model=None, device="cpu")
        self._req_q = req_q
        self._res_q = res_q
        self._worker_id = worker_id

    def curate_batch(self, atoms_list):  # pragma: no cover - server-side only
        raise NotImplementedError("Curation runs in the GPU server process.")

    def _infer(self, atoms_to_compute) -> tuple[np.ndarray, np.ndarray]:
        positions = np.ascontiguousarray(
            np.stack([a.positions for a in atoms_to_compute]), dtype=np.float32
        )
        self._req_q.put((_KIND_FORCES, self._worker_id, positions))
        reply = self._res_q.get()
        if isinstance(reply, _ServerError):
            raise RuntimeError(f"GPU force server failed:\n{reply.message}")
        return reply  # (forces, energy), already post-processed by the server

    def clean_up(self) -> None:  # no local model to clean up
        pass


def _worker_main(worker_id, pdb_path, n_systems, fn, seed, req_q, res_q) -> None:
    """Entry point for a worker process: build a local MultiAtoms and run ``fn``."""
    import torch

    torch.set_num_threads(1)  # CPU-bound integrator; avoid thread oversubscription
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        manager = RemoteModelManager(req_q, res_q, worker_id)
        multi = MultiAtoms(
            template=pdb_path, model_manager=manager, n_systems=n_systems
        )
        result = fn(multi, worker_id)
    except Exception:
        req_q.put((_KIND_DONE, worker_id, _WorkerError(traceback.format_exc())))
    else:
        req_q.put((_KIND_DONE, worker_id, result))


class PolyAtoms:
    """Run ``workers`` parallel ``MultiAtoms`` simulations sharing one GPU.

    Args:
        pdb_path: Template structure. A single path is used by every worker; a
            list of length ``workers`` gives each worker its own template, so
            different workers can simulate different systems on one shared GPU.
        model_manager: A live ``ModelManager`` on the GPU; stays in the main
            process and serves all workers. Owned by ``PolyAtoms`` -- its
            ``clean_up()`` runs on context-manager exit. It must handle every
            worker's system (ML potentials are general over molecules).
        n_systems: Systems per worker. A single int applies to every worker; a
            list of length ``workers`` sets each worker's count independently --
            size it so each worker's batched forward costs comparable GPU time
            (bigger systems -> fewer replicas).
        workers: Number of worker processes K. ``None`` runs a single in-main
            ``MultiAtoms`` (no IPC, no overlap); ``1`` uses the real pool with
            one worker (useful to measure IPC overhead).
    """

    def __init__(
        self,
        pdb_path: Path | str | List[Path | str],
        model_manager: ModelManager,
        n_systems: int | List[int] = 1,
        workers: Optional[int] = 2,
    ):
        self._pdb_path = pdb_path
        self._model_manager = model_manager
        self._n_systems = n_systems
        self._workers = workers

    def __enter__(self) -> "PolyAtoms":
        return self

    def __exit__(self, *exc) -> bool:
        self._model_manager.clean_up()
        return False

    @staticmethod
    def _per_worker(value, k, what):
        """Broadcast a scalar to ``k`` workers, or check a per-worker list length."""
        if isinstance(value, (list, tuple)):
            if len(value) != k:
                raise ValueError(
                    f"need one {what} per worker; got {len(value)} for {k} workers"
                )
            return list(value)
        return [value] * k

    def run(
        self,
        fn: Callable[[MultiAtoms, int], Any],
        *,
        seeds: Optional[List[int]] = None,
    ) -> List[Any]:
        """Run ``fn(multi, worker_id)`` in each worker; return one result per worker.

        ``fn`` must be a top-level function (or ``functools.partial`` of one) so
        standard pickle can ship it to the workers. ``seeds`` (one per worker,
        defaults to ``range(workers)``) seed each worker's RNG so their
        trajectories diverge.
        """
        if self._workers is None:
            multi = MultiAtoms(
                template=self._per_worker(self._pdb_path, 1, "template")[0],
                model_manager=self._model_manager,
                n_systems=self._per_worker(self._n_systems, 1, "n_systems")[0],
            )
            return [fn(multi, 0)]

        k = self._workers
        if seeds is None:
            seeds = list(range(k))
        elif len(seeds) != k:
            raise ValueError(
                f"need one seed per worker; got {len(seeds)} for {k} workers"
            )

        templates = self._per_worker(self._pdb_path, k, "template")
        n_systems = self._per_worker(self._n_systems, k, "n_systems")

        ctx = mp.get_context("spawn")
        req_q = ctx.Queue()
        res_qs = [ctx.Queue() for _ in range(k)]

        # One reusable atoms list per worker (cell/pbc/numbers intact); the
        # server overwrites only positions per request, so each worker is
        # curated against its own template even when the workers differ.
        loaded = [ase.io.read(t) for t in templates]
        views_per_worker = [
            [loaded[w].copy() for _ in range(n_systems[w])] for w in range(k)
        ]

        procs = [
            ctx.Process(
                target=_worker_main,
                args=(i, templates[i], n_systems[i], fn, seeds[i], req_q,
                      res_qs[i]),
            )
            for i in range(k)
        ]
        for p in procs:
            p.start()

        results: List[Any] = [None] * k
        try:
            self._serve(k, req_q, res_qs, views_per_worker, results, procs)
        finally:
            # A hard death aborts the run with survivors still blocked on a
            # force reply; terminate them so nothing is left orphaned.
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()

        self._raise_on_worker_error(results)
        return results

    def _serve(self, k, req_q, res_qs, views_per_worker, results, procs) -> None:
        """Main-process server loop: answer force requests until all K workers done.

        Waits on the request queue with a timeout so a worker that dies hard
        (segfault / OOM / kill) without sending ``_KIND_DONE`` is noticed
        instead of hanging the server forever. On such a death we fail fast:
        the surviving workers share the same model and GPU, so they are almost
        certainly doomed too and continuing would only burn compute.

        Detection is deferred, not instantaneous. Each worker is synchronous --
        it sends one force request then blocks on its reply -- so a live worker
        keeps the queue busy and the ``Empty`` timeout (where the liveness check
        runs) only fires once the queue drains, i.e. once every surviving worker
        has finished or is blocked. The run therefore never hangs, but if other
        workers are still doing useful steps when one crashes, the abort happens
        when they wind down rather than the instant of the crash.
        """
        done = [False] * k
        remaining = k
        while remaining:
            try:
                kind, worker_id, payload = req_q.get(timeout=_POLL_TIMEOUT_S)
            except queue.Empty:
                # No message for a while -- check whether a worker crashed. A
                # cleanly finished worker always enqueues its DONE before
                # exiting, so if the queue is idle any dead worker died hard.
                self._check_for_dead_worker(procs, done)
                continue

            if kind == _KIND_DONE:
                results[worker_id] = payload
                done[worker_id] = True
                remaining -= 1
                continue

            # kind == _KIND_FORCES: payload is positions (m, n_atoms, 3), curated
            # against this worker's own template (workers may differ in system).
            views = views_per_worker[worker_id]
            m = payload.shape[0]
            for i in range(m):
                views[i].set_positions(payload[i])
            try:
                forces, energy = self._model_manager._infer(views[:m])
            except Exception:
                res_qs[worker_id].put(_ServerError(traceback.format_exc()))
            else:
                res_qs[worker_id].put((forces, energy))

    @staticmethod
    def _check_for_dead_worker(procs, done) -> None:
        """Raise if a worker exited without reporting a result (hard crash)."""
        for i, p in enumerate(procs):
            if not done[i] and p.exitcode is not None:
                raise RuntimeError(
                    f"PolyAtoms worker {i} died without returning a result "
                    f"(exit code {p.exitcode}); aborting the run."
                )

    @staticmethod
    def _raise_on_worker_error(results) -> None:
        errors = [
            (i, r.message) for i, r in enumerate(results) if isinstance(r, _WorkerError)
        ]
        if errors:
            detail = "\n\n".join(f"--- worker {i} ---\n{m}" for i, m in errors)
            raise RuntimeError(f"PolyAtoms worker(s) failed:\n{detail}")


__all__ = ["PolyAtoms", "RemoteModelManager"]
