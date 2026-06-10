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
_STAGES = ("get", "rebuild", "curate_h2d", "forward", "d2h_post", "put")


def server_main(
    req_q, res_qs, spec: ModelSpec, z: np.ndarray, ready_evt, start_evt, profile_q
) -> None:
    """GPU server, instrumented to expose the Tier-1 headroom.

    Times each per-request stage (queue/unpickle, rebuild, curate+H2D, forward,
    D2H+post, pickle/put). Accumulators reset when ``start_evt`` fires, so warmup
    requests are excluded and the breakdown reflects steady state. The profile
    (stage sums + forward count) is sent back on ``profile_q`` at shutdown.
    """
    manager = build_manager(spec, _device())
    z = np.asarray(z)
    acc = dict.fromkeys(_STAGES, 0.0)
    n_fwd = 0
    timed = False
    ready_evt.set()

    while True:
        t = time.perf_counter()
        item = req_q.get()
        get_dt = time.perf_counter() - t
        if item is None:  # poison pill -> send profile and shut down
            break
        if not timed and start_evt.is_set():  # drop warmup; measure steady state
            acc = dict.fromkeys(_STAGES, 0.0)
            n_fwd = 0
            timed = True

        worker_id, positions = item
        try:
            t = time.perf_counter()
            views = [_AtomsView(positions[i], z) for i in range(positions.shape[0])]
            rebuild_dt = time.perf_counter() - t

            t = time.perf_counter()
            batched = manager.curate_batch(views)
            _sync()
            curate_dt = time.perf_counter() - t

            t = time.perf_counter()
            energy_raw, forces_raw = manager.model_forward(batched)
            _sync()
            forward_dt = time.perf_counter() - t

            t = time.perf_counter()
            energy = energy_raw.flatten().detach().cpu().numpy()
            forces = forces_raw.detach().cpu().numpy()
            forces, energy = manager.post_process_hook(forces, energy)
            d2h_dt = time.perf_counter() - t

            t = time.perf_counter()
            res_qs[worker_id].put((forces, energy))
            put_dt = time.perf_counter() - t
        except Exception as exc:  # forward the error so the worker doesn't hang
            res_qs[worker_id].put(_ServerError(repr(exc)))
            continue

        acc["get"] += get_dt
        acc["rebuild"] += rebuild_dt
        acc["curate_h2d"] += curate_dt
        acc["forward"] += forward_dt
        acc["d2h_post"] += d2h_dt
        acc["put"] += put_dt
        n_fwd += 1

    profile_q.put((acc, n_fwd))
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
    readied = False
    try:
        manager = RemoteModelManager(req_q, res_q, worker_id)
        multi, integrators = setup_simulation(manager, pdb_path, n_systems, temp)
        if warmup:
            run_md(multi, integrators, warmup)  # also warms the server
        ready_q.put(worker_id)
        readied = True

        start_evt.wait()
        t_start = time.time()
        run_md(multi, integrators, n_steps)
        t_end = time.time()
        multi.clean_up()
        done_q.put((worker_id, t_start, t_end, n_systems * n_steps))
    except Exception:  # e.g. server OOM propagated as RuntimeError -- unblock main
        if not readied:
            ready_q.put(worker_id)
        done_q.put((worker_id, 0.0, 0.0, -1))  # steps=-1 marks failure


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


def run_force_server(
    ctx, spec, pdb_path, z, args, n_systems, workers
) -> tuple[float, list, tuple]:
    """Run ``workers`` x n_systems against the instrumented server.

    Returns (wall, results, profile). ``wall`` is the concurrent production span,
    or -1.0 if any worker failed (e.g. CUDA OOM at large n_systems).
    """
    k = workers
    req_q = ctx.Queue()
    res_qs = [ctx.Queue() for _ in range(k)]
    ready_q = ctx.Queue()
    done_q = ctx.Queue()
    profile_q = ctx.Queue()
    start_evt = ctx.Event()
    server_ready = ctx.Event()

    server = ctx.Process(
        target=server_main,
        args=(req_q, res_qs, spec, z, server_ready, start_evt, profile_q),
    )
    server.start()
    server_ready.wait()

    workers = [
        ctx.Process(
            target=worker_main,
            args=(i, req_q, res_qs[i], pdb_path, n_systems, args.n_steps,
                  args.warmup_steps, args.temperature_k, ready_q, done_q, start_evt),
        )
        for i in range(k)
    ]
    for w in workers:
        w.start()

    for _ in range(k):  # wait until every worker has finished warmup (or failed)
        ready_q.get()

    start_evt.set()  # release the timed run
    results = [done_q.get() for _ in range(k)]

    req_q.put(None)  # poison pill -> server sends profile and shuts down
    profile = profile_q.get()
    server.join()
    for w in workers:
        w.join()

    if any(r[3] == -1 for r in results):  # a worker failed (e.g. OOM)
        return -1.0, results, profile

    t_start = min(r[1] for r in results)
    t_end = max(r[2] for r in results)
    return t_end - t_start, results, profile


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
    parser.add_argument(
        "--sweep", default=None,
        help="Comma-separated n-systems values; run the server headroom probe for "
        "each and print a summary table (skips the single-process baseline).",
    )
    parser.add_argument(
        "--workers-sweep", default=None,
        help="Comma-separated worker counts K; with --sweep, runs the full "
        "(K x n-systems) grid. Defaults to just --workers.",
    )
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

    if args.sweep:
        sizes = [int(x) for x in args.sweep.split(",")]
        workers_list = (
            [int(x) for x in args.workers_sweep.split(",")]
            if args.workers_sweep else [args.workers]
        )
        run_sweep(ctx, spec, str(pdb_path), z, args, sizes, workers_list)
        tmpdir.cleanup()
        return

    print(f"Phase A: baseline (single process), {args.n_systems} systems ...")
    base_elapsed = run_baseline(ctx, spec, str(pdb_path), args)

    print(f"Phase B: force server, K={args.workers} workers x "
          f"{args.n_systems} systems ...")
    fs_wall, results, profile = run_force_server(
        ctx, spec, str(pdb_path), z, args, args.n_systems, args.workers
    )

    tmpdir.cleanup()
    report(args, base_elapsed, fs_wall, results, profile)


def summarize(profile) -> dict | None:
    """Reduce the server stage accumulators to fractions + the Tier-1 ceiling.

    Partitions the per-request cycle into forward / ipc (get+put, the pickle) /
    rest (rebuild+H2D+D2H). ``ceiling`` = 1/forward = the max Tier-1 speedup if
    every non-forward stage could be fully overlapped with compute.
    """
    acc, n_fwd = profile
    total = sum(acc.values())
    if not n_fwd or total <= 0:
        return None
    fwd = acc["forward"]
    ipc = acc["get"] + acc["put"]
    return {
        "ms_req": total / n_fwd * 1e3,
        "fwd": fwd / total,
        "ipc": ipc / total,
        "rest": (total - fwd - ipc) / total,
        "ceiling": total / fwd if fwd > 0 else float("inf"),
        "acc": acc,
        "n_fwd": n_fwd,
        "total": total,
    }


def _print_headroom(s: dict) -> None:
    acc, total, n_fwd = s["acc"], s["total"], s["n_fwd"]
    labels = {
        "get": "get+unpickle", "rebuild": "rebuild views", "curate_h2d": "curate+H2D",
        "forward": "model_forward (GPU)", "d2h_post": "D2H+post", "put": "pickle+put",
    }
    print(f"  SERVER per-request breakdown ({n_fwd} forwards, "
          f"{s['ms_req']:.2f} ms/req):")
    for stg in _STAGES:
        print(f"    {labels[stg]:<22} {acc[stg] / total * 100:5.1f}%  "
              f"{acc[stg] / n_fwd * 1e3:7.3f} ms")
    print(f"  Tier-1 headroom (overlap all non-forward w/ compute): "
          f"~{s['ceiling']:.2f}x   (forward = {s['fwd'] * 100:.0f}% of cycle)")
    lever = ("pickle dominates -> shared-memory transport"
             if s["ipc"] >= s["rest"]
             else "transfer/prep dominates -> pinned + CUDA streams (Tier 1)")
    print(f"    non-forward: ipc {s['ipc'] * 100:.0f}% (get+put)  vs  "
          f"rest {s['rest'] * 100:.0f}% (rebuild+H2D+D2H)  ->  {lever}")


def report(args, base_elapsed, fs_wall, results, profile) -> None:
    k = args.workers
    print("=" * 70)
    print("Force-server proof-of-concept")
    print("=" * 70)
    print(f"  model        {args.model}   systems/worker {args.n_systems}   "
          f"atoms {args.n_atoms}   steps {args.n_steps}")
    print(f"  workers (K)  {k}")
    print("-" * 70)
    if fs_wall < 0:
        print("  force server: ABORTED (worker error, likely CUDA OOM).")
    else:
        base_steps = args.n_systems * args.n_steps
        base_tput = base_steps / base_elapsed
        fs_tput = k * base_steps / fs_wall
        speedup = fs_tput / base_tput
        print(f"  baseline (1 proc):  {base_elapsed:7.2f} s   "
              f"{base_tput:10.0f} system-steps/s")
        print(f"  force server (K={k}): {fs_wall:7.2f} s   "
              f"{fs_tput:10.0f} system-steps/s")
        print(f"  SPEEDUP            {speedup:5.2f}x   "
              f"(parallel efficiency {speedup / k * 100:4.0f}% of K)")
        per = "  ".join(f"w{r[0]}:{r[2] - r[1]:.1f}s" for r in sorted(results))
        print(f"  per-worker wall    {per}")
    s = summarize(profile)
    if s:
        print("-" * 70)
        _print_headroom(s)
    print("=" * 70)


def run_sweep(ctx, spec, pdb_path, z, args, sizes, workers_list) -> None:
    """Run the server headroom probe over a (workers x n-systems) grid."""
    print("=" * 84)
    print(f"Server headroom sweep | model {args.model} | atoms {args.n_atoms} | "
          f"K={workers_list} | steps {args.n_steps}")
    print("=" * 84)
    hdr = (f"{'K':>3} {'n_sys':>6} {'ms/req':>8} {'fwd%':>6} {'ceiling':>9} "
           f"{'ipc%':>6} {'rest%':>6} {'Msteps/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k in workers_list:
        for ns in sizes:
            wall, results, profile = run_force_server(
                ctx, spec, pdb_path, z, args, ns, k
            )
            s = summarize(profile)
            if wall < 0 or s is None:
                print(f"{k:>3} {ns:>6}  aborted (likely OOM); next K.")
                break
            tput = k * ns * args.n_steps / wall / 1e6
            print(f"{k:>3} {ns:>6} {s['ms_req']:>8.2f} {s['fwd'] * 100:>5.1f} "
                  f"{s['ceiling']:>8.2f}x {s['ipc'] * 100:>5.0f} "
                  f"{s['rest'] * 100:>5.0f} {tput:>9.3f}")
        print("-" * len(hdr))
    print("ceiling = 1/fwd% = max Tier-1 speedup if ALL non-forward overlaps compute.")
    print("ipc = get+put (pickle -> shared memory).  rest = rebuild+H2D+D2H "
          "(-> CUDA streams / pipeline).")
    print("Server memory ~ one n_systems batch, independent of K -> same OOM ceiling.")
    print("Msteps/s is the *instrumented* throughput (per-stage syncs add overhead).")
    print("NOTE: 'get' includes any wait for the next request; saturated (K>=2) it")
    print("~= unpickle. K=1 or small n_systems -> server starved -> ipc% inflated.")


if __name__ == "__main__":
    main()
