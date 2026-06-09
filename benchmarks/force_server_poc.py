#!/usr/bin/env python
"""Proof-of-concept: a multi-process GPU "force server" for multiatoms.

The duty-cycle measurement (benchmarks/measure_duty_cycle.py) showed the GPU
sits idle ~45% of the time while the ASE integrators do their (single-threaded,
GIL-bound) Langevin stepping. This POC reclaims that idle time by running K
independent worker processes -- each an ordinary ``MultiAtoms`` simulation --
that ship their force requests to one central GPU process. While worker A does
its CPU integration, the GPU is busy with worker B's batch.

Design (deliberately minimal):

  * The greenlet scheduler / BatchedAtoms / ProxyCalculator are reused *as-is*
    via ``MultiAtoms`` -- no package edits. Each worker is the existing
    single-process design, unchanged, except its ModelManager is remote.
  * ``RemoteModelManager`` is the only new piece on the worker side: instead of
    running the model locally it ships positions to the server and blocks for
    the result, then distributes exactly as the base class does.
  * ``server_main`` owns the model and the single CUDA context. It pulls
    requests off one shared queue (natural round-robin by arrival) and runs one
    batched forward at a time.
  * Workers are CPU-only (they never import torch.cuda), so there is exactly one
    CUDA context. Multiprocessing uses 'spawn' (fork + CUDA is unsafe).

It runs two phases and prints the speedup:
  Phase A -- baseline: the current single-process design (local GPU model).
  Phase B -- force server: K workers sharing one GPU server.

IMPORTANT: the workers are CPU-bound, so the overlap only materializes if the
node actually has >= K CPU cores free. On SLURM request --cpus-per-task >= K+1.

    pixi run -e bench python benchmarks/force_server_poc.py \
        --model schnet --n-systems 256 --n-atoms 60 --n-steps 200 --workers 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from ase import units
from ase.io import read as ase_read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from multiatoms import ModelManager, MultiAtoms
from multiatoms.ase_md import NullLogger, SmartLangevin

# Make the sibling bench module importable both when run directly and when
# re-imported by 'spawn'ed child processes (sys.path[0] is this directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact models/managers from the duty-cycle bench so the POC measures
# the same workload (and so there is one source of truth for the SchNet config).
from measure_duty_cycle import (  # noqa: E402
    _MODEL_DEFAULTS,
    BenchSchNet,
    SchNetManager,
    SyntheticManager,
    SyntheticPotential,
    _template_pdb,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Model spec + builders (server-side only; workers never build a model)
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """Everything the server needs to reconstruct the model. Picklable."""

    model: str
    hidden: int
    n_layers: int
    n_gaussians: int
    cutoff: float


def build_manager(spec: ModelSpec, device: str) -> ModelManager:
    """Construct the real ModelManager + model on ``device`` (GPU process)."""
    if spec.model == "schnet":
        model = BenchSchNet(
            hidden=spec.hidden,
            n_interactions=spec.n_layers,
            n_gaussians=spec.n_gaussians,
            cutoff=spec.cutoff,
        )
        manager_cls = SchNetManager
    else:
        model = SyntheticPotential(hidden=spec.hidden, n_layers=spec.n_layers)
        manager_cls = SyntheticManager
    return manager_cls(model=model.to(device).eval(), device=device)


def run_inference(manager: ModelManager, views: list) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ModelManager.compute_energy_and_forces steps 2-5 (no distribute).

    Kept here rather than refactored into the package so the POC stays
    non-invasive; the real integration would expose this as ModelManager._infer.
    """
    batched = manager.curate_batch(views)
    energy_raw, forces_raw = manager.model_forward(batched)
    energy = energy_raw.flatten().detach().cpu().numpy()
    forces = forces_raw.detach().cpu().numpy()
    return manager.post_process_hook(forces, energy)


class _AtomsView:
    """Minimal stand-in for the atoms attributes curate_batch reads.

    The full design would send the static topology (z, cell) once; here z is
    fixed across requests, so the server holds it and only positions travel.
    """

    __slots__ = ("positions", "_z")

    def __init__(self, positions: np.ndarray, z: np.ndarray):
        self.positions = positions
        self._z = z

    def __len__(self) -> int:
        return len(self._z)

    def get_atomic_numbers(self) -> np.ndarray:
        return self._z


# --------------------------------------------------------------------------- #
# Worker-side: remote ModelManager (the only new worker code)
# --------------------------------------------------------------------------- #
class RemoteModelManager(ModelManager):
    """Ships force requests to the GPU server instead of running a model.

    Overrides only ``compute_energy_and_forces`` -- the scheduler, BatchedAtoms
    and ProxyCalculator all drive it exactly as they drive the local manager.
    The per-system position cache and result distribution are unchanged.
    """

    def __init__(self, req_q, res_q, worker_id: int):
        super().__init__(model=None, device="cpu")
        self._req_q = req_q
        self._res_q = res_q
        self._worker_id = worker_id

    def curate_batch(self, atoms_list):  # pragma: no cover - never called remotely
        raise NotImplementedError("RemoteModelManager curates on the server.")

    def compute_energy_and_forces(self, atoms_list) -> None:
        # Same per-system cache filter as the base class.
        to_compute = [
            a
            for a in atoms_list
            if a._cached_positions is None
            or not np.array_equal(a._cached_positions, a.positions)
        ]
        if not to_compute:
            return

        positions = np.ascontiguousarray(
            np.stack([a.positions for a in to_compute]), dtype=np.float32
        )
        self._req_q.put((self._worker_id, positions))
        reply = self._res_q.get()
        if isinstance(reply, _ServerError):
            raise RuntimeError(f"force server failed: {reply.message}")
        forces, energy = reply

        self.distribute_results(to_compute, forces, energy)
        for atom in to_compute:
            atom._cached_positions = atom.positions.copy()

    def cleanup(self) -> None:  # no local model to clean up
        pass


@dataclass
class _ServerError:
    message: str


# --------------------------------------------------------------------------- #
# Shared simulation setup (used by baseline and workers -> identical workload)
# --------------------------------------------------------------------------- #
def setup_simulation(manager: ModelManager, pdb_path: str, n_systems: int, temp: float):
    multi = MultiAtoms(
        pdb_path=pdb_path, model_manager=manager, n_systems=n_systems
    )
    multi.foreach(
        lambda a: MaxwellBoltzmannDistribution(a, temperature_K=temp), multi.atoms
    )
    integrators = multi.map(
        lambda a: SmartLangevin(
            a,
            timestep=1 * units.fs,
            temperature_K=temp,
            friction=0.01 / units.fs,
            logfile=NullLogger(),
        ),
        multi.atoms,
    )
    return multi, integrators


def run_md(multi: MultiAtoms, integrators, n_steps: int) -> None:
    with multi.parallel():
        multi.foreach(lambda integ: integ.run(n_steps), integrators)


# --------------------------------------------------------------------------- #
# Process entry points (top-level so 'spawn' can import them)
# --------------------------------------------------------------------------- #
def server_main(req_q, res_qs, spec: ModelSpec, z: np.ndarray, ready_evt) -> None:
    manager = build_manager(spec, _device())
    z = np.asarray(z)
    ready_evt.set()
    while True:
        item = req_q.get()
        if item is None:  # poison pill -> shut down
            break
        worker_id, positions = item
        try:
            views = [_AtomsView(positions[i], z) for i in range(positions.shape[0])]
            forces, energy = run_inference(manager, views)
            res_qs[worker_id].put((forces, energy))
        except Exception as exc:  # forward the error so the worker doesn't hang
            res_qs[worker_id].put(_ServerError(repr(exc)))
    manager.cleanup()


def baseline_main(spec, pdb_path, n_systems, n_steps, warmup, temp, out_q) -> None:
    """Phase A: the current single-process design with a local GPU model."""
    manager = build_manager(spec, _device())
    multi, integrators = setup_simulation(manager, pdb_path, n_systems, temp)
    if warmup:
        run_md(multi, integrators, warmup)
    _sync()
    t0 = time.perf_counter()
    run_md(multi, integrators, n_steps)
    _sync()
    out_q.put(time.perf_counter() - t0)
    multi.clean_up()


def worker_main(
    worker_id, req_q, res_q, pdb_path, n_systems, n_steps, warmup, temp, ready_q,
    done_q, start_evt,
) -> None:
    torch.set_num_threads(1)  # CPU-bound integrator; avoid thread oversubscription
    manager = RemoteModelManager(req_q, res_q, worker_id)
    multi, integrators = setup_simulation(manager, pdb_path, n_systems, temp)
    if warmup:
        run_md(multi, integrators, warmup)  # also warms the server (real requests)
    ready_q.put(worker_id)

    start_evt.wait()
    t_start = time.time()
    run_md(multi, integrators, n_steps)
    t_end = time.time()
    multi.clean_up()
    done_q.put((worker_id, t_start, t_end, n_systems * n_steps))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_baseline(ctx, spec, pdb_path, args) -> float:
    out_q = ctx.Queue()
    p = ctx.Process(
        target=baseline_main,
        args=(spec, pdb_path, args.n_systems, args.n_steps, args.warmup_steps,
              args.temperature_k, out_q),
    )
    p.start()
    elapsed = out_q.get()
    p.join()
    return elapsed


def run_force_server(ctx, spec, pdb_path, z, args) -> tuple[float, list]:
    k = args.workers
    req_q = ctx.Queue()
    res_qs = [ctx.Queue() for _ in range(k)]
    ready_q = ctx.Queue()
    done_q = ctx.Queue()
    start_evt = ctx.Event()
    server_ready = ctx.Event()

    server = ctx.Process(
        target=server_main, args=(req_q, res_qs, spec, z, server_ready)
    )
    server.start()
    server_ready.wait()

    workers = [
        ctx.Process(
            target=worker_main,
            args=(i, req_q, res_qs[i], pdb_path, args.n_systems, args.n_steps,
                  args.warmup_steps, args.temperature_k, ready_q, done_q, start_evt),
        )
        for i in range(k)
    ]
    for w in workers:
        w.start()

    for _ in range(k):  # wait until every worker has finished warmup
        ready_q.get()

    start_evt.set()  # release the timed run
    results = [done_q.get() for _ in range(k)]

    req_q.put(None)  # poison pill -> server shuts down
    server.join()
    for w in workers:
        w.join()

    # True concurrent wall span across all workers.
    t_start = min(r[1] for r in results)
    t_end = max(r[2] for r in results)
    wall = t_end - t_start
    return wall, results


def main() -> None:
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["schnet", "synthetic"], default="schnet")
    parser.add_argument("--n-systems", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2, help="K worker processes")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--n-atoms", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-gaussians", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=10.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA. This POC is meant for a GPU node; on CPU there is "
            "no GPU to keep busy and the speedup is meaningless.\n"
        )

    # Fill model knobs from the shared defaults (CLI value wins).
    for knob, value in _MODEL_DEFAULTS[args.model].items():
        if getattr(args, knob, None) is None:
            setattr(args, knob, value)
    spec = ModelSpec(args.model, args.hidden, args.n_layers, args.n_gaussians,
                     args.cutoff)

    # One shared template structure for every process.
    pdb_path, tmpdir = _template_pdb(args)
    template = ase_read(pdb_path)
    z = template.get_atomic_numbers()

    ctx = mp.get_context("spawn")

    print(f"Phase A: baseline (single process), {args.n_systems} systems ...")
    base_elapsed = run_baseline(ctx, spec, str(pdb_path), args)

    print(f"Phase B: force server, K={args.workers} workers x "
          f"{args.n_systems} systems ...")
    fs_wall, results = run_force_server(ctx, spec, str(pdb_path), z, args)

    tmpdir.cleanup()
    report(args, base_elapsed, fs_wall, results)


def report(args, base_elapsed, fs_wall, results) -> None:
    k = args.workers
    base_steps = args.n_systems * args.n_steps
    fs_steps = k * base_steps
    base_tput = base_steps / base_elapsed  # system-steps / s
    fs_tput = fs_steps / fs_wall
    speedup = fs_tput / base_tput

    print("=" * 70)
    print("Force-server proof-of-concept")
    print("=" * 70)
    print(f"  model        {args.model}   systems/worker {args.n_systems}   "
          f"atoms {args.n_atoms}   steps {args.n_steps}")
    print(f"  workers (K)  {k}")
    print("-" * 70)
    print(f"  baseline (1 proc):  {base_elapsed:7.2f} s   "
          f"{base_tput:10.0f} system-steps/s")
    print(f"  force server (K={k}): {fs_wall:7.2f} s   "
          f"{fs_tput:10.0f} system-steps/s")
    print("-" * 70)
    print(f"  SPEEDUP            {speedup:5.2f}x   "
          f"(parallel efficiency {speedup / k * 100:4.0f}% of K)")
    per = "  ".join(f"w{r[0]}:{r[2] - r[1]:.1f}s" for r in sorted(results))
    print(f"  per-worker wall    {per}")
    print("=" * 70)
    if speedup < 1.05:
        print("No gain -- likely IPC-bound or too few free CPU cores "
              "(need >= K cores). Check --cpus-per-task.")
    elif speedup >= 1.5:
        print(f"Clear win: ~{speedup:.1f}x more sampling throughput on one GPU.")
    else:
        print("Modest gain -- try more workers (K) or check CPU-core availability.")


if __name__ == "__main__":
    main()
