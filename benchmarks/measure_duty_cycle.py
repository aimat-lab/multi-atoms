#!/usr/bin/env python
"""Measure the GPU duty cycle of a parallel multiatoms MD run.

Why
---
The current scheduler alternates synchronously between the GPU (one batched
forward pass for all systems) and the CPU (every integrator does its Langevin
step). While the integrators step, the GPU is idle. The proposed multi-process
"force server" reclaims that idle time by overlapping one worker group's CPU
integration with another group's GPU forward pass.

The ceiling on that idea is *exactly* the fraction of wall-clock time the GPU is
currently idle. This script measures it, so we know whether the overlap buys
~1.1x or ~3x before building any of the multi-process machinery.

It reports two ceilings:

  * optimistic  -- based on pure GPU kernel time (``model_forward``). Reachable
    only if batch curation + host<->device transfers could also be pipelined.
  * conservative -- based on the whole ``compute_energy_and_forces`` critical
    section, which is what the GPU server process would run serially per
    request. Only the integrator stepping is clearly overlappable.

For each: max throughput speedup ~= 1 / duty_cycle, and it takes about
1 / duty_cycle worker processes to saturate the GPU.

Run on a machine with a GPU (a CPU run only validates the harness; the ratio is
not representative):

    # Synthetic model. Tune --hidden / --n-layers until the reported GPU
    # forward time roughly matches your real model, and --n-atoms to match your
    # system size:
    pixi run python benchmarks/measure_duty_cycle.py --n-systems 256 --n-steps 200

    # Or point it at your real model + structure. --manager-factory names a
    # "module.path:function" that takes a device string and returns a
    # configured ModelManager:
    pixi run python benchmarks/measure_duty_cycle.py \
        --manager-factory mypkg.bench:build_manager --pdb system.pdb \
        --n-systems 256 --n-steps 200
"""

from __future__ import annotations

import argparse
import importlib
import math
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor, nn

from multiatoms import ModelManager, MultiAtoms
from multiatoms.ase_md import NullLogger, SmartLangevin
from multiatoms.model_manager import ModelManager as _MM

try:
    from ase import units
except ImportError:  # pragma: no cover - ase always present
    units = None


# --------------------------------------------------------------------------- #
# Profiler
# --------------------------------------------------------------------------- #
class DutyCycleProfiler:
    """Times the GPU critical section of a parallel run by wrapping a manager.

    Monkeypatches a ``ModelManager`` instance (no package edits): ``model_forward``
    is timed with a CUDA sync on both sides to capture true kernel time, and
    ``compute_energy_and_forces`` is timed end-to-end to capture the full
    GPU-process critical section (curation + forward + transfers + distribute).
    """

    def __init__(self, model_manager: _MM, device: str):
        self.device = device
        self._cuda = device.startswith("cuda")
        self.forward_s: list[float] = []
        self.cef_s: list[float] = []
        self._wrap(model_manager)

    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize()

    def _wrap(self, mm: _MM) -> None:
        orig_forward = mm.model_forward
        orig_cef = mm.compute_energy_and_forces

        def timed_forward(batched_input):
            self._sync()
            t0 = time.perf_counter()
            out = orig_forward(batched_input)
            self._sync()
            self.forward_s.append(time.perf_counter() - t0)
            return out

        def timed_cef(atoms_list):
            t0 = time.perf_counter()
            out = orig_cef(atoms_list)
            # cef has already synced internally via timed_forward + .cpu()
            self.cef_s.append(time.perf_counter() - t0)
            return out

        # Instance-level overrides; the inner ``self.model_forward`` call inside
        # the original cef resolves to this patched one (instance dict wins).
        mm.model_forward = timed_forward
        mm.compute_energy_and_forces = timed_cef

    def reset(self) -> None:
        self.forward_s.clear()
        self.cef_s.clear()


# --------------------------------------------------------------------------- #
# Synthetic GPU-bound model (stand-in for a real ML potential)
# --------------------------------------------------------------------------- #
class SyntheticPotential(nn.Module):
    """A tunable per-atom MLP whose forward + autograd grad burns GPU time.

    Not physically meaningful -- it only needs to (a) cost a realistic amount of
    GPU time and (b) match the ``model(**input)`` + ``get_forces`` contract.
    """

    def __init__(self, hidden: int = 512, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(3, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.head = nn.Linear(hidden, 1)

    def forward(self, pos: Tensor, batch_idx: Tensor, **_: Tensor) -> Tensor:
        x = torch.tanh(self.embed(pos))
        for block in self.blocks:
            x = torch.tanh(block(x))
        per_atom_e = self.head(x).squeeze(-1)
        n_sys = int(batch_idx.max().item()) + 1
        energy = torch.zeros(n_sys, device=pos.device, dtype=pos.dtype)
        energy = energy.index_add(0, batch_idx, per_atom_e)
        return energy

    def get_forces(self, energy: Tensor, pos: Tensor) -> Tensor:
        (grad,) = torch.autograd.grad(energy.sum(), pos, create_graph=False)
        return -grad

    def cleanup(self) -> None:
        pass


class SyntheticManager(ModelManager):
    """ModelManager for SyntheticPotential: one system per batch index."""

    def curate_batch(self, atoms_list: List) -> dict[str, Tensor]:
        n_atoms = len(atoms_list[0])
        positions = np.stack([a.positions for a in atoms_list]).reshape(-1, 3)
        pos = torch.tensor(positions, dtype=torch.float32, device=self.device)
        batch_idx = torch.arange(
            len(atoms_list), device=self.device
        ).repeat_interleave(n_atoms)
        return {"pos": pos, "batch_idx": batch_idx}


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def build_synthetic(args, device: str) -> tuple[_MM, Path, tempfile.TemporaryDirectory]:
    """Synthetic manager + a temp PDB of ``--n-atoms`` carbon atoms in a box."""
    rng = np.random.default_rng(0)
    box = max(5.0, args.n_atoms ** (1 / 3) * 2.0)
    positions = rng.uniform(0.0, box, size=(args.n_atoms, 3))
    template = Atoms("C" + str(args.n_atoms), positions=positions)
    template.set_cell([box, box, box])

    tmpdir = tempfile.TemporaryDirectory(prefix="duty_cycle_")
    pdb_path = Path(tmpdir.name) / "synthetic.pdb"
    ase_write(pdb_path, template, format="proteindatabank")

    model = SyntheticPotential(hidden=args.hidden, n_layers=args.n_layers)
    model = model.to(device).eval()
    manager = SyntheticManager(model=model, device=device)
    return manager, pdb_path, tmpdir


def build_real(args, device: str) -> _MM:
    module_path, _, func_name = args.manager_factory.partition(":")
    if not func_name:
        raise ValueError("--manager-factory must be 'module.path:function'")
    module = importlib.import_module(module_path)
    factory = getattr(module, func_name)
    manager = factory(device)
    if not isinstance(manager, ModelManager):
        raise TypeError(f"{args.manager_factory} did not return a ModelManager")
    return manager


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_parallel_md(multi: MultiAtoms, integrators, n_steps: int) -> float:
    """Run ``n_steps`` of batched parallel MD; return wall-clock seconds."""
    t0 = time.perf_counter()
    with multi.parallel():
        multi.foreach(lambda integ: integ.run(n_steps), integrators)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-systems", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    # synthetic-model knobs (ignored when --manager-factory is given)
    parser.add_argument("--n-atoms", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    # real-model hook
    parser.add_argument("--manager-factory", default=None, help="module.path:function")
    parser.add_argument("--pdb", default=None, help="structure for the real model")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not device.startswith("cuda"):
        print(
            "WARNING: running on CPU. This validates the harness but the "
            "GPU/CPU time ratio is NOT representative of a real GPU run.\n"
        )

    tmpdir = None
    if args.manager_factory:
        if not args.pdb:
            parser.error("--pdb is required with --manager-factory")
        manager = build_real(args, device)
        pdb_path = Path(args.pdb)
        mode = f"real ({args.manager_factory})"
    else:
        manager, pdb_path, tmpdir = build_synthetic(args, device)
        mode = f"synthetic (hidden={args.hidden}, n_layers={args.n_layers})"

    multi = MultiAtoms(
        pdb_path=pdb_path, model_manager=manager, n_systems=args.n_systems
    )
    n_atoms = len(multi.atoms[0])

    # Per-system setup (serial, no GPU): velocities + integrators.
    fs = units.fs if units is not None else 1.0
    multi.foreach(
        lambda a: MaxwellBoltzmannDistribution(a, temperature_K=args.temperature_k),
        multi.atoms,
    )
    integrators = multi.map(
        lambda a: SmartLangevin(
            a,
            timestep=1 * fs,
            temperature_K=args.temperature_k,
            friction=0.01 / fs,
            logfile=NullLogger(),
        ),
        multi.atoms,
    )

    profiler = DutyCycleProfiler(manager, device)

    # Warmup: lets CUDA do lazy init / allocator warmup / autotune so the timed
    # run measures steady state, not first-call overhead.
    if args.warmup_steps > 0:
        run_parallel_md(multi, integrators, args.warmup_steps)
        profiler.reset()

    total_wall = run_parallel_md(multi, integrators, args.n_steps)

    multi.clean_up()
    if tmpdir is not None:
        tmpdir.cleanup()

    report(args, mode, device, n_atoms, total_wall, profiler)


def report(
    args, mode, device, n_atoms, total_wall, profiler: DutyCycleProfiler
) -> None:
    n_rounds = len(profiler.cef_s)
    n_forwards = len(profiler.forward_s)
    sum_forward = sum(profiler.forward_s)
    sum_cef = sum(profiler.cef_s)
    prep = sum_cef - sum_forward  # curation + transfers + distribute
    integrator = total_wall - sum_cef  # pure worker-side CPU stepping

    # Guard against the degenerate no-rounds case.
    if n_rounds == 0 or total_wall <= 0:
        print("No forward passes recorded -- nothing to report.")
        return

    d_forward = sum_forward / total_wall  # optimistic duty cycle
    d_server = sum_cef / total_wall  # conservative duty cycle

    def block(name: str, duty: float) -> str:
        duty = min(max(duty, 1e-9), 1.0)
        speedup = 1.0 / duty
        k = math.ceil(speedup)
        return (
            f"  {name:<13} duty={duty * 100:5.1f}%   "
            f"max speedup~={speedup:4.2f}x   workers to saturate K~={k}"
        )

    print("=" * 70)
    print("GPU duty-cycle measurement")
    print("=" * 70)
    print(f"  mode          {mode}")
    print(f"  device        {device}")
    print(f"  systems       {args.n_systems}   atoms/system {n_atoms}")
    print(f"  rounds {n_rounds}   forwards {n_forwards}   steps {args.n_steps}")
    print("-" * 70)
    print("Wall-clock breakdown:")
    print(f"  total                 {total_wall * 1e3:9.1f} ms")
    print(
        f"  GPU forward           {sum_forward * 1e3:9.1f} ms"
        f"   ({sum_forward / total_wall * 100:4.1f}%)   "
        f"{sum_forward / max(n_forwards, 1) * 1e3:6.2f} ms/forward"
    )
    print(
        f"  prep+transfer+dist    {prep * 1e3:9.1f} ms"
        f"   ({prep / total_wall * 100:4.1f}%)"
    )
    print(
        f"  integrator stepping   {integrator * 1e3:9.1f} ms"
        f"   ({integrator / total_wall * 100:4.1f}%)"
    )
    print("-" * 70)
    print("Overlap ceilings (how much the multi-process force server can buy):")
    print(block("optimistic", d_forward))
    print(block("conservative", d_server))
    print("=" * 70)
    if d_server > 0.85:
        print("Verdict: GPU is already well-utilized -- the overlap buys little.")
    elif d_server < 0.55:
        print("Verdict: lots of idle GPU time -- the force server is worth building.")
    else:
        print("Verdict: moderate idle time -- worth a prototype to confirm the gain.")


if __name__ == "__main__":
    main()
