# multiatoms

Parallel, GPU-batched molecular dynamics on top of [ASE](https://wiki.fysik.dtu.dk/ase/) `Atoms`.

`multiatoms` lets you run many MD simulations of the same system at once and
batch their model evaluations into a single GPU forward pass. Each simulation is
an ordinary ASE `Atoms` object driven by an ordinary ASE integrator (Langevin,
Velocity Verlet, BFGS, ...). A cooperative greenlet scheduler pauses every
simulation when it needs forces, collects all the pending systems, runs **one**
batched forward pass, and hands the results back — so GPU utilization scales with
the number of parallel systems instead of being dominated by per-call overhead.

## Why

ML potentials are fast per atom but small systems underfill the GPU. Stepping
`N` copies in lockstep and batching their force evaluations turns `N` tiny
forward passes into one big one, which is where the throughput comes from.

## Install

With [pixi](https://pixi.sh) (recommended for development):

```bash
pixi install
pixi run test
```

Or as a dependency of another project (editable path dep):

```toml
# pixi.toml
[pypi-dependencies]
multiatoms = { path = "../multi-atoms", editable = true }
```

Or straight from git:

```bash
pip install "multiatoms @ git+https://github.com/aimat-lab/multi-atoms.git"
```

Runtime dependencies: `ase`, `numpy`, `greenlet`, `torch`.

## Concepts

| Class          | Role                                                                              |
| -------------- | -------------------------------------------------------------------------------- |
| `MultiAtoms`   | Owns `n_systems` `BatchedAtoms` and exposes `map` / `foreach` / `parallel()`.     |
| `BatchedAtoms` | An ASE `Atoms` subclass whose `get_forces()` yields to the scheduler in parallel. |
| `ModelManager` | Abstract base you subclass to turn a batch of systems into model inputs/outputs.  |
| `HubScheduler` | Trampoline (star-topology) greenlet scheduler; O(1) stack depth regardless of N.  |

## Usage

You implement one method — `curate_batch` — to describe how a list of systems
becomes a batched model input. Everything else (caching, scheduling, result
distribution) is handled for you.

```python
import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from multiatoms import MultiAtoms, ModelManager
from multiatoms.ase_md import NullLogger, SmartLangevin


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

    def model_forward(self, batched_input):
        pos = batched_input["pos"]
        pos.requires_grad_(True)
        with torch.set_grad_enabled(True):
            energy = self.model(pos, batched_input["batch_idx"])
            forces = self.model.get_forces(energy, pos)
        return energy, forces

    # Optional: convert/scale units before results are distributed.
    def post_process_hook(self, forces, energy):
        return forces, energy


device = "cuda" if torch.cuda.is_available() else "cpu"
manager = MyModelManager(model=my_model.to(device).eval(), device=device)

# `template` accepts an ASE Atoms object or a path to any ASE-readable file
# (PDB, xyz, CIF, ...); its cell, PBC and constraints are copied into each system.
multi = MultiAtoms(template="system.pdb", model_manager=manager, n_systems=64)

# Per-system setup runs serially (no GPU calls here):
multi.foreach(lambda a: MaxwellBoltzmannDistribution(a, temperature_K=300), multi.atoms)
integrators = multi.map(
    lambda a: SmartLangevin(a, timestep=1 * units.fs, temperature_K=300,
                            friction=0.01 / units.fs, logfile=NullLogger()),
    multi.atoms,
)

# Inside parallel(), get_forces() across all systems is batched into one GPU pass:
with multi.parallel():
    multi.foreach(lambda integrator: integrator.run(1000), integrators)

multi.clean_up()
```

### The `ModelManager` contract

- **`curate_batch(atoms_list) -> dict[str, Tensor]`** *(required)* — build the
  batched model input from the systems that need new forces.
- **`model_forward(batched_input) -> (energy, forces)`** *(optional)* — defaults
  to `model(**batched_input)` + `model.get_forces(energy, pos)`. Override for a
  custom calling convention.
- **`post_process_hook(forces, energy) -> (forces, energy)`** *(optional)* — unit
  conversion / scaling before distribution. Defaults to identity.
- **`clean_up()`** *(optional)* — defaults to calling `model.clean_up()` if present.

Forces and energies must come back in ASE units (eV / eV·Å⁻¹); positions handed
to `curate_batch` are in Å.

### `map` / `foreach` / `parallel()`

- `map(fn, *iterables)` / `foreach(fn, *iterables)` apply `fn` across systems.
  Outside `parallel()` they run serially; inside, each system runs in its own
  greenlet and force evaluations are batched.
- Only call code that triggers `get_forces()` (i.e. integrator steps) inside
  `parallel()`. Set-up like attaching integrators or loggers should happen
  outside it.

### Running many parallel integrators (file-descriptor fix)

ASE's `MolecularDynamics` opens `/dev/null` whenever no `logfile` is given, so
running many integrators at once leaks one file descriptor each and can exhaust
the process limit. The optional `multiatoms.ase_md` module fixes this:

- `NullLogger` — a no-op stream; pass it as `logfile=` to any ASE
  integrator/optimizer (`Langevin`, `VelocityVerlet`, `BFGS`, ...).
- `SmartLangevin` — a `Langevin` subclass that returns a `NullLogger` instead of
  opening `/dev/null`, so the fix applies even when ASE forces `logfile=None`
  internally.

These live in a separate module and are never imported by the package core —
`multiatoms` itself stays agnostic of how you drive dynamics.

## Development

```bash
pixi run test     # pytest
pixi run lint     # ruff check
pixi run format   # ruff format
```

## License

MIT — see [LICENSE](LICENSE).
