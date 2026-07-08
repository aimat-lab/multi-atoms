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

    def clean_up(self) -> None:
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


def test_heterogeneous_templates_and_n_systems():
    """Different template AND different n_systems per worker on one server.

    Worker 0 gets 3 systems of 4 atoms; worker 1 gets 2 systems of 6 atoms.
    The differing atom counts would crash a single shared ``views`` list
    (``set_positions`` shape mismatch), so passing proves the server curates
    each worker against its own template.
    """
    tpl4 = str(_write_template(4))
    tpl6 = str(_write_template(6))
    with PolyAtoms(
        [tpl4, tpl6], _make_manager(), n_systems=[3, 2], workers=2
    ) as poly:
        results = poly.run(simulate, seeds=[0, 1])
    assert len(results) == 2
    assert [p.shape for p in results[0]] == [(4, 3)] * 3
    assert [p.shape for p in results[1]] == [(6, 3)] * 2
    assert np.isfinite(np.concatenate(results[0] + results[1])).all()


def test_heterogeneous_worker_matches_homogeneous():
    """A worker's trajectory is independent of what the other worker simulates.

    Worker 0 (template A) inside a heterogeneous [A, B] pool must reproduce the
    same bit-for-bit trajectory as the same worker run alone -- proving the
    server serves each worker from its own template with no cross-talk.
    """
    tpl_a = str(_write_template(4))
    tpl_b = str(_write_template(6))
    with PolyAtoms(
        [tpl_a, tpl_b], _make_manager(), n_systems=[2, 3], workers=2
    ) as poly:
        hetero = poly.run(simulate, seeds=[0, 1])
    with PolyAtoms(tpl_a, _make_manager(), n_systems=2, workers=1) as poly:
        ref = poly.run(simulate, seeds=[0])[0]
    for a, b in zip(hetero[0], ref):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_template_count_mismatch_raises():
    tpl_a = str(_write_template(4))
    with PolyAtoms([tpl_a], _make_manager(), n_systems=2, workers=2) as poly:
        with pytest.raises(ValueError, match="one template per worker"):
            poly.run(simulate, seeds=[0, 1])


def test_n_systems_count_mismatch_raises(template_pdb):
    mgr = _make_manager()
    with PolyAtoms(template_pdb, mgr, n_systems=[2, 3, 4], workers=2) as poly:
        with pytest.raises(ValueError, match="one n_systems per worker"):
            poly.run(simulate, seeds=[0, 1])


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
