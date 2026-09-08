# Usage

You implement one method, `curate_batch`, to describe how a list of systems
becomes a batched model input. Everything else (caching, scheduling, result
distribution) is handled for you.

```python
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from multiatoms import MultiAtoms, ModelManager
from multiatoms.ase_md import FdSafeLangevin, NullLogger


class MyModelManager(ModelManager):
    """Turn a batch of systems into one model call."""

    def curate_batch(self, atoms_list):
        n_systems = len(atoms_list)
        n_atoms = len(atoms_list[0])
        pos = torch.tensor(
            np.stack([a.positions for a in atoms_list]).reshape(-1, 3),
            dtype=torch.float32, device=self.device,
        )
        batch_idx = torch.arange(n_systems, device=self.device).repeat_interleave(n_atoms)
        return {"pos": pos, "batch_idx": batch_idx}


device = "cuda" if torch.cuda.is_available() else "cpu"
manager = MyModelManager(model=my_model.to(device).eval(), device=device)

# `template` accepts an ASE Atoms object or a path to any ASE-readable file
# (PDB, xyz, CIF, ...); its cell, PBC and constraints are copied into each system.
multi = MultiAtoms(template="system.pdb", model_manager=manager, n_systems=64)

# Per-system setup runs serially (no GPU calls here):
multi.foreach(lambda a: MaxwellBoltzmannDistribution(a, temperature_K=300), multi.atoms)
integrators = multi.map(
    lambda a: FdSafeLangevin(a, timestep=1 * units.fs, temperature_K=300,
                             friction=0.01 / units.fs, logfile=NullLogger()),
    multi.atoms,
)

# Inside parallel(), get_forces() across all systems is batched into one GPU pass:
with multi.parallel():
    multi.foreach(lambda integrator: integrator.run(1000), integrators)

multi.clean_up()
```

## The `ModelManager` contract

- **`curate_batch(atoms_list) -> Any`** *(required)*: build the batched model
  input, in whatever form your `model_forward` consumes (a dict of tensors, a
  PyG `Batch`, ...). `atoms_list` holds only the systems whose positions changed
  since the last batch, so its length varies from step to step and is generally
  not `n_systems`, so take the count from `len(atoms_list)`.
- **`model_forward(batched_input) -> (energy, forces)`** *(optional)*: defaults
  to `model(**batched_input)` + `model.get_forces(energy, pos)`, which assumes
  your batch is a dict holding a key literally named `"pos"` and that the model
  exposes `get_forces(energy, pos)`. Most real MLIPs match none of that, so
  expect to override it.
- **`post_process_hook(forces, energy) -> (forces, energy)`** *(optional)*: unit
  conversion / scaling before distribution. Defaults to identity.
- **`clean_up()`** *(optional)*: defaults to calling `model.clean_up()` if present.

Forces and energies must come back in ASE units (eV / eV·Å⁻¹); positions handed
to `curate_batch` are in Å.

Two rules are load-bearing. Neither is checked, and breaking either produces
wrong forces rather than an error:

- **`model_forward` must return forces in system-major order**, matching the
  order `curate_batch` received the systems. Results are distributed by
  fixed-stride slicing of the flat `(Σ atoms, 3)` array, which cannot detect any
  other layout. A manager that regroups atoms by element looks correct and
  mis-assigns every force.
- **Every system has the same atom count and ordering.** That holds by
  construction (all systems are copies of one template) and `ProxyCalculator`
  fixes the count when it is built, so changing a system's atom count afterwards
  misaligns every later slice.

To exercise a manager on its own, checking a batched forward against a stock
single-system calculator for instance, call **`infer(atoms_list) -> (forces, energy)`**.
It runs one batch through curation, forward and post-processing with no caching
and no result distribution.

The `device` you pass to `ModelManager` is stored and handed to your
`curate_batch`; multiatoms never resolves, validates or acts on it, and never
moves your model. Whether the run is actually on the GPU is therefore entirely
determined by your own `device` string and your own `model.to(device)`. Log it
yourself if you want a record. (A `torch` build that does not match the driver
makes `torch.cuda.is_available()` return `False` silently, and nothing in the
stack warns about it.)

## What batching does not cover

What batching *cannot* collapse is anything your `curate_batch` does per system
on the CPU, so how much you gain depends on where your model builds its graph:

- **Built inside the forward**, from positions plus a `batch` vector (SchNet via
  `torch_cluster.radius_graph`, and most models that take raw coordinates): the
  graph for all `N` systems is built in one GPU call. Batching applies to the
  whole step and the speedup is large, as plotted in the
  [README](../README.md) figure.
- **Required as input** (MACE, NequIP and other models that expect a prebuilt
  `edge_index` / cell shifts): the graph must exist before the model is called,
  so it is built in your `curate_batch`. Written the obvious way, as a Python
  loop calling an ASE-style neighbour list once per system, that part stays
  serial and caps the gain, no matter how well the forward batches. If per-step
  cost is dominated by graph construction rather than the forward, expect a
  modest speedup, not the plotted one.

In the second case the lever is your `curate_batch`, not multiatoms: build the
radius graph once for the whole batch on the GPU where the model allows it, use
a fast periodic neighbour builder (`vesin`, `matscipy`) where it does not, or
reuse neighbour lists across steps with a skin buffer.

## `map` / `foreach` / `parallel()`

- `map(fn, *iterables)` / `foreach(fn, *iterables)` apply `fn` across systems.
  Outside `parallel()` they run serially; inside, each system runs in its own
  greenlet and force evaluations are batched.
- Only call code that triggers `get_forces()` (i.e. integrator steps) inside
  `parallel()`. Set-up like attaching integrators or loggers should happen
  outside it.

## `PolyAtoms`: sharing one GPU across processes

A single `MultiAtoms` run alternates between the GPU (one batched forward) and the
CPU (every integrator steps), so the GPU sits idle a good fraction of the time.
`PolyAtoms` reclaims it: it runs `workers` independent `MultiAtoms` simulations in
separate processes that ship their force requests to one shared GPU server in the
main process. While one worker integrates on the CPU, the GPU serves another's
batch (267 → 423 ns/day, ~1.6×, on one A100 with `workers=2`).

Note what is and is not overlapped. Workers send **positions**; the server does
curation *and* the forward. So `PolyAtoms` overlaps a worker's integrator stepping
with the server's *(curate + forward)*. It does **not** spread graph building
across processes. If your `curate_batch` is expensive (the "required as input"
case above), the server serialises it for every worker and becomes the bottleneck,
and adding workers will not help. It pays off when the server is dominated by the
GPU forward, which is the regime the [README](../README.md) figure was
measured in.

```python
from multiatoms import PolyAtoms

def simulate(multi, worker_id):          # top-level so `spawn` can pickle it
    integrators = multi.map(lambda a: FdSafeLangevin(a, ...), multi.atoms)
    with multi.parallel():
        multi.foreach(lambda i: i.run(1000), integrators)
    return multi.get_positions()

if __name__ == "__main__":               # required for spawn
    with PolyAtoms("system.pdb", manager, n_systems=64, workers=2) as poly:
        results = poly.run(simulate, seeds=[0, 1])   # one result per worker
```

`template` takes the same forms as `MultiAtoms`: an ASE `Atoms` object or a path
to any ASE-readable file. Both it and `n_systems` also accept a per-worker list,
so different systems can share the same GPU server; size each worker's count so
its batched forward costs comparable GPU time (bigger systems → fewer replicas):

```python
with PolyAtoms(["ligand_a.pdb", ligand_b_atoms], manager,
               n_systems=[64, 32], workers=2) as poly:
    results = poly.run(simulate, seeds=[0, 1])
```

## Running many parallel integrators (file-descriptor fix)

ASE's `MolecularDynamics` opens `/dev/null` whenever no `logfile` is given, so
running many integrators at once leaks one file descriptor each and can exhaust
the process limit. The optional `multiatoms.ase_md` module fixes this:

- `NullLogger`: a no-op stream; pass it as `logfile=` to any ASE
  integrator/optimizer (`Langevin`, `VelocityVerlet`, `BFGS`, ...).
- `FdSafeLangevin`: a `Langevin` subclass that returns a `NullLogger` instead of
  opening `/dev/null`, so the fix applies even when ASE forces `logfile=None`
  internally.

These live in a separate module and are never imported by the package core.
`multiatoms` itself stays agnostic of how you drive dynamics.
