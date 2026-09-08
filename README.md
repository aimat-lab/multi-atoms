<p align="center">
  <img src="docs/logo.png" alt="multiatoms" width="420">
</p>

<p align="center">
  <a href="https://github.com/aimat-lab/multi-atoms/actions/workflows/ci.yml">
    <img src="https://github.com/aimat-lab/multi-atoms/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

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

What batching *cannot* collapse is anything your `curate_batch` does per system
on the CPU — so how much you gain depends on where your model builds its graph:

- **Built inside the forward**, from positions plus a `batch` vector (SchNet via
  `torch_cluster.radius_graph`, and most models that take raw coordinates): the
  graph for all `N` systems is built in one GPU call. Batching applies to the
  whole step and the speedup is large — the figure above.
- **Required as input** (MACE, NequIP and other models that expect a prebuilt
  `edge_index` / cell shifts): the graph must exist before the model is called,
  so it is built in your `curate_batch`. Written the obvious way — a Python loop
  calling an ASE-style neighbour list once per system — that part stays serial
  and caps the gain, no matter how well the forward batches. If per-step cost is
  dominated by graph construction rather than the forward, expect a modest
  speedup, not the one plotted above.

In the second case the lever is your `curate_batch`, not multiatoms: build the
radius graph once for the whole batch on the GPU where the model allows it, use
a fast periodic neighbour builder (`vesin`, `matscipy`) where it does not, or
reuse neighbour lists across steps with a skin buffer.

![MD throughput scaling on an A100](docs/throughput_scaling.png)

*MD throughput on one A100 (SchNet, alanine dipeptide, no solvent). multiatoms takes raw ASE
from 8.3 → 267 ns/day single-process (32×), and the PolyAtoms worker pool reaches
~423 ns/day (51×) — within ~12% of [mlcg](https://github.com/ClementiGroup/mlcg),
a fully GPU-native code, while every simulation stays a standard ASE object driven
by a standard ASE integrator.*

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
| `PolyAtoms`    | Runs `workers` `MultiAtoms` in separate processes sharing one GPU force server.   |

## Quickstart

You implement one method — `curate_batch` — to describe how a list of systems
becomes a batched model input. Everything else (caching, scheduling, result
distribution) is handled for you. Full walkthrough in
[docs/usage.md](docs/usage.md).

```python
multi = MultiAtoms(template="system.pdb", model_manager=manager, n_systems=64)
integrators = multi.map(lambda a: Langevin(a, ...), multi.atoms)

# inside parallel(), get_forces() across all systems is one batched GPU pass
with multi.parallel():
    multi.foreach(lambda i: i.run(1000), integrators)
```

## Development

```bash
pixi run test     # pytest
pixi run lint     # ruff check
pixi run format   # ruff format
```

## License

MIT — see [LICENSE](LICENSE).
