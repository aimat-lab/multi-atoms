"""Tests for PolyAtoms: the multi-process GPU force server.

Run on CPU with 'spawn'. Everything passed across the process boundary
(``fn``, the manager factory below) is defined at module top level so standard
pickle can ship it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write as ase_write
from ase.md.verlet import VelocityVerlet
from torch import Tensor

from multiatoms import ModelManager, MultiAtoms, PolyAtoms
from multiatoms.ase_md import NullLogger


class HarmonicModel:
    """Forces = -positions (restoring toward origin); energy = sum of positions."""

    def __call__(self, pos: Tensor, batch_idx: Tensor) -> Tensor:
        n = int(batch_idx.max().item()) + 1 if len(batch_idx) else 0
        return torch.stack([pos[batch_idx == b].sum() for b in range(n)])

    def get_forces(self, energy: Tensor, pos: Tensor) -> Tensor:
        return -pos

    def cleanup(self) -> None:
        pass


class HarmonicManager(ModelManager):
    def curate_batch(self, atoms_list: List) -> dict[str, Tensor]:
        n_atoms = len(atoms_list[0])
        pos = torch.tensor(
            np.stack([a.positions for a in atoms_list]).reshape(-1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        batch_idx = torch.arange(len(atoms_list)).repeat_interleave(n_atoms)
        return {"pos": pos, "batch_idx": batch_idx}


def _make_manager() -> HarmonicManager:
    return HarmonicManager(model=HarmonicModel(), device="cpu")


def _write_template(n_atoms: int = 4) -> Path:
    rng = np.random.default_rng(0)
    atoms = Atoms("C" + str(n_atoms), positions=rng.uniform(0, 5, size=(n_atoms, 3)))
    atoms.set_cell([10, 10, 10])
    tmp = Path(tempfile.mkdtemp(prefix="poly_test_")) / "template.pdb"
    ase_write(tmp, atoms, format="proteindatabank")
    return tmp


def simulate(multi: MultiAtoms, worker_id: int) -> list:
    """Top-level so spawn can pickle it. Runs a few parallel Verlet steps."""
    integrators = multi.map(
        lambda a: VelocityVerlet(a, timestep=0.5, logfile=NullLogger()), multi.atoms
    )
    with multi.parallel():
        multi.foreach(lambda integ: integ.run(3), integrators)
    return [p.copy() for p in multi.get_positions()]


@pytest.fixture
def template_pdb():
    return str(_write_template())


def test_workers_none_runs_in_main(template_pdb):
    """workers=None -> single in-main MultiAtoms, one result."""
    with PolyAtoms(template_pdb, _make_manager(), n_systems=3, workers=None) as poly:
        results = poly.run(simulate)
    assert len(results) == 1
    assert len(results[0]) == 3  # n_systems position arrays
    assert np.isfinite(np.concatenate(results[0])).all()


def test_pool_runs_one_result_per_worker(template_pdb):
    """workers=2 -> two workers, two results, forces correctly served over IPC."""
    with PolyAtoms(template_pdb, _make_manager(), n_systems=3, workers=2) as poly:
        results = poly.run(simulate, seeds=[0, 1])
    assert len(results) == 2
    for per_worker in results:
        assert len(per_worker) == 3
        assert np.isfinite(np.concatenate(per_worker)).all()


def test_pool_matches_in_main(template_pdb):
    """Remote (pooled) trajectory must match the local one bit-for-bit.

    VelocityVerlet is deterministic and Verlet starts from rest, so worker 0's
    result must equal the in-main result for the same template.
    """
    with PolyAtoms(template_pdb, _make_manager(), n_systems=2, workers=None) as poly:
        local = poly.run(simulate)[0]
    with PolyAtoms(template_pdb, _make_manager(), n_systems=2, workers=1) as poly:
        remote = poly.run(simulate, seeds=[0])[0]
    for a, b in zip(local, remote):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_seed_count_mismatch_raises(template_pdb):
    with PolyAtoms(template_pdb, _make_manager(), n_systems=2, workers=2) as poly:
        with pytest.raises(ValueError, match="one seed per worker"):
            poly.run(simulate, seeds=[0])


def hard_crash(multi: MultiAtoms, worker_id: int) -> list:
    """Top-level so spawn can pickle it. Dies hard without sending _KIND_DONE."""
    import os

    os._exit(1)


def test_hard_worker_death_raises_not_hangs(template_pdb, monkeypatch):
    """A worker that dies without reporting must fail the run, not hang it."""
    from multiatoms import poly_atoms

    # Shrink the liveness poll so the test does not wait the full timeout.
    monkeypatch.setattr(poly_atoms, "_POLL_TIMEOUT_S", 0.2)

    with PolyAtoms(template_pdb, _make_manager(), n_systems=2, workers=1) as poly:
        with pytest.raises(RuntimeError, match="died without returning a result"):
            poly.run(hard_crash, seeds=[0])
