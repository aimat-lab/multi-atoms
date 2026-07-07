"""Integration tests for the multiatoms module.

These tests verify the full pipeline works correctly with a dummy model.
"""

from typing import List

import numpy as np
import pytest
import torch
from greenlet import greenlet
from torch import Tensor

from multiatoms.core import (
    BatchedAtoms,
    MultiAtomAttribute,
)
from multiatoms.model_manager import (
    ModelManager,
)
from multiatoms.proxy_calculator import (
    ProxyCalculator,
)
from multiatoms.scheduler import HubScheduler


class DummyModel:
    """Dummy ML model with predictable outputs for testing.

    Energy: sum of all position components
    Forces: negative of positions (simple harmonic-like restoring force)
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self.forward_count = 0

    @property
    def device(self):
        return self._device

    def __call__(self, pos, batch_idx, partial_charges=None):
        self.forward_count += 1

        # Energy = sum of positions for each batch
        n_batches = batch_idx.max().item() + 1 if len(batch_idx) > 0 else 0
        energies = []
        for b in range(n_batches):
            mask = batch_idx == b
            energies.append(pos[mask].sum())

        return torch.stack(energies) if energies else torch.tensor([])

    def get_forces(self, energy, pos):
        # Forces = -positions (restoring force toward origin)
        return -pos

    def cleanup(self):
        pass


class DummyModelManager(ModelManager):
    """ModelManager with dummy scaling for testing."""

    def __init__(self, model, device: str = "cpu"):
        super().__init__(model, device)
        # Scales simulate nm->Å conversion (factor of 10)
        self.energy_scale = 10.0
        self.force_scale = 10.0

    def curate_batch(self, atoms_list: List) -> dict[str, Tensor]:
        n_batches = len(atoms_list)
        n_atoms = len(atoms_list[0])

        positions_np = np.stack([a.positions for a in atoms_list])
        positions_np = positions_np.reshape(-1, 3) * 0.1  # A -> nm
        pos = torch.tensor(positions_np, dtype=torch.float32, device=self.device)

        batch_idx = torch.arange(n_batches, device=self.device).repeat_interleave(
            n_atoms
        )

        return {
            "pos": pos,
            "batch_idx": batch_idx,
        }

    def post_process_hook(
        self, forces: np.ndarray, energy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply scaling to simulate unit conversion."""
        energy = energy * self.energy_scale
        forces = forces * self.force_scale
        return forces, energy


class TestBatchedAtomsIntegration:
    """Integration tests for BatchedAtoms with scheduler."""

    def test_get_forces_serial_mode(self):
        """Test get_forces in serial mode calls GPU directly."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 3
        scheduler = HubScheduler(model_manager)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        atom = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H", "H"],
            positions=positions,
        )
        atom.calc = ProxyCalculator(n_atoms=n_atoms)

        # Serial mode (default)
        assert atom._parallel_mode is False

        forces = atom.get_forces()

        # Forces should be -positions * 0.1 (Å->nm) * 10 (postprocess) = -positions
        expected_forces = -positions
        np.testing.assert_allclose(forces, expected_forces, rtol=1e-5)

    def test_get_forces_parallel_mode_yields(self):
        """Test get_forces in parallel mode yields to scheduler."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        scheduler = HubScheduler(model_manager)

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        atom = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H"],
            positions=positions,
        )
        atom.calc = ProxyCalculator(n_atoms=n_atoms)
        atom._parallel_mode = True

        # Track execution
        yielded = []
        resumed = []

        def worker():
            yielded.append(True)
            forces = atom.get_forces()
            resumed.append(True)
            return forces

        g = greenlet(worker)
        scheduler.set_greenlet_pool([g])
        scheduler.kick_off()

        assert len(yielded) == 1
        assert len(resumed) == 1


class TestFullPipelineIntegration:
    """End-to-end integration tests simulating parallel MD steps."""

    def test_single_system_multiple_steps(self):
        """Test single system running multiple force evaluations."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 3
        scheduler = HubScheduler(model_manager)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

        atom = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H", "H"],
            positions=positions.copy(),
        )
        atom.calc = ProxyCalculator(n_atoms=n_atoms)

        # Simulate 3 MD steps
        for step in range(3):
            forces = atom.get_forces()
            # Update positions (simple Euler step)
            atom.positions = atom.positions + 0.01 * forces

        # Model should have been called 3 times (once per step)
        assert model.forward_count == 3

    def test_parallel_systems_batched(self):
        """Test multiple systems running in parallel with batched GPU calls."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        n_systems = 4
        scheduler = HubScheduler(model_manager)

        # Create multiple atoms
        atoms_list = []
        for i in range(n_systems):
            positions = np.array([[float(i), 0.0, 0.0], [0.0, float(i), 0.0]])
            atom = BatchedAtoms(
                model_manager=model_manager,
                scheduler=scheduler,
                symbols=["H", "H"],
                positions=positions,
            )
            atom.calc = ProxyCalculator(n_atoms=n_atoms)
            atom._parallel_mode = True
            atoms_list.append(atom)

        # Simulate MD step: each atom calls get_forces once
        n_steps = 3
        results = [[] for _ in range(n_systems)]

        def make_worker(atom_idx):
            def worker():
                for step in range(n_steps):
                    forces = atoms_list[atom_idx].get_forces()
                    results[atom_idx].append(forces.copy())
                    # Simple position update
                    atoms_list[atom_idx].positions = (
                        atoms_list[atom_idx].positions + 0.01 * forces
                    )

            return worker

        greenlets = [greenlet(make_worker(i)) for i in range(n_systems)]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # All systems should have completed all steps
        for i in range(n_systems):
            assert len(results[i]) == n_steps

        # Should have n_steps batched forward calls (all systems batched together)
        assert model.forward_count == n_steps

    def test_parallel_systems_different_positions(self):
        """Verify each system gets its own correct forces."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        n_systems = 3
        scheduler = HubScheduler(model_manager)

        atoms_list = []
        initial_positions = []
        for i in range(n_systems):
            # Each system has different positions
            positions = np.array([[float(i + 1), 0.0, 0.0], [0.0, float(i + 1), 0.0]])
            initial_positions.append(positions.copy())

            atom = BatchedAtoms(
                model_manager=model_manager,
                scheduler=scheduler,
                symbols=["H", "H"],
                positions=positions,
            )
            atom.calc = ProxyCalculator(n_atoms=n_atoms)
            atom._parallel_mode = True
            atoms_list.append(atom)

        collected_forces = [None] * n_systems

        def make_worker(idx):
            def worker():
                collected_forces[idx] = atoms_list[idx].get_forces()

            return worker

        greenlets = [greenlet(make_worker(i)) for i in range(n_systems)]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # Verify each system got correct forces (forces = -positions)
        for i in range(n_systems):
            expected = -initial_positions[i]
            np.testing.assert_allclose(collected_forces[i], expected, rtol=1e-5)

    def test_heterogeneous_completion_no_stale_forces(self):
        """Systems that finish at different times must never read stale forces.

        Regression test for the scheduler round-boundary bug: when the last
        greenlet in the pool finishes mid-round, the systems that already
        yielded that round used to resume with forces from the *previous*
        round. With forces = -positions, any force read that does not match the
        system's current positions is a stale read -> we catch it directly.

        The trailing greenlet is given the *fewest* evaluations so it dies at
        the wrap position, which is exactly the condition that triggered the bug.
        """
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        scheduler = HubScheduler(model_manager)

        def build(seed):
            rng = np.random.default_rng(seed)
            atom = BatchedAtoms(
                model_manager=model_manager,
                scheduler=scheduler,
                symbols=["H", "H"],
                positions=rng.uniform(0.0, 3.0, size=(n_atoms, 3)),
            )
            atom.calc = ProxyCalculator(n_atoms=n_atoms)
            atom._parallel_mode = True
            return atom

        # (atom, n_force_evals) — earlier system runs longer, trailing one dies first.
        specs = [(build(0), 3), (build(1), 2)]
        violations = []

        def make_worker(atom, n_evals, idx):
            def worker():
                for step in range(n_evals):
                    forces = atom.get_forces()
                    if not np.allclose(forces, -atom.get_positions(), atol=1e-4):
                        violations.append((idx, step))
                    atom.set_positions(atom.get_positions() + 0.05 * forces)

            return worker

        greenlets = [
            greenlet(make_worker(atom, n, i)) for i, (atom, n) in enumerate(specs)
        ]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        assert violations == [], f"systems read stale forces at {violations}"


class TestMultiAtomAttribute:
    """Tests for MultiAtomAttribute proxy class."""

    def test_getitem_returns_list(self):
        """Test __getitem__ returns values from all atoms."""

        # Create mock atoms with arrays
        class MockAtom:
            def __init__(self, val):
                self.arrays = {"charges": np.array([val, val * 2])}

        atoms_list = [MockAtom(1.0), MockAtom(2.0), MockAtom(3.0)]
        proxy = MultiAtomAttribute(atoms_list, "arrays")

        result = proxy["charges"]

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [2.0, 4.0])
        np.testing.assert_array_equal(result[2], [3.0, 6.0])

    def test_setitem_sets_on_all_atoms(self):
        """Test __setitem__ sets values on all atoms."""

        class MockAtom:
            def __init__(self):
                self.arrays = {"charges": None}

        atoms_list = [MockAtom(), MockAtom(), MockAtom()]
        proxy = MultiAtomAttribute(atoms_list, "arrays")

        new_charges = np.array([0.5, -0.5])
        proxy["charges"] = new_charges

        for atom in atoms_list:
            np.testing.assert_array_equal(atom.arrays["charges"], new_charges)

    def test_call_invokes_on_all_atoms(self):
        """Test __call__ invokes method on all atoms."""

        class MockAtom:
            def __init__(self, val):
                self.val = val

            def get_value(self, multiplier=1):
                return self.val * multiplier

        atoms_list = [MockAtom(1), MockAtom(2), MockAtom(3)]
        proxy = MultiAtomAttribute(atoms_list, "get_value")

        result = proxy(multiplier=2)

        assert result == [2, 4, 6]


class TestProxyCalculator:
    """Tests for ProxyCalculator."""

    def test_set_results_stores_references(self):
        calc = ProxyCalculator(n_atoms=2)

        forces = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        energy = np.array([100.0, 200.0])

        calc.set_results(forces, energy, atom_index=1)

        # Should slice correctly for atom_index=1
        expected_forces = forces[2:4]  # [7,8,9], [10,11,12]
        np.testing.assert_array_equal(calc.get_forces(), expected_forces)
        assert calc.get_potential_energy() == 200.0

    def test_get_forces_without_results_raises(self):
        calc = ProxyCalculator(n_atoms=1)

        with pytest.raises(RuntimeError, match="Forces not set"):
            calc.get_forces()

    def test_get_energy_without_results_raises(self):
        calc = ProxyCalculator(n_atoms=1)

        with pytest.raises(RuntimeError, match="Energy not set"):
            calc.get_potential_energy()

    def test_calculate_populates_results_dict(self):
        calc = ProxyCalculator(n_atoms=2)

        forces = np.array([[1, 2, 3], [4, 5, 6]])
        energy = np.array([42.0])

        calc.set_results(forces, energy, atom_index=0)
        calc.calculate()

        assert calc.results["energy"] == 42.0
        np.testing.assert_array_equal(calc.results["forces"], forces)


class TestPositionCaching:
    """Tests for position caching behavior."""

    def test_cache_prevents_redundant_computation(self):
        """Test that unchanged positions don't trigger recomputation."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        scheduler = HubScheduler(model_manager)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        atom = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H"],
            positions=positions.copy(),
        )
        atom.calc = ProxyCalculator(n_atoms=n_atoms)

        # First call - should compute
        atom.get_forces()
        assert model.forward_count == 1

        # Second call with same positions - should use cache
        atom.get_forces()
        assert model.forward_count == 1

        # Change positions - should recompute
        atom.positions = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        atom.get_forces()
        assert model.forward_count == 2

    def test_parallel_caching_per_system(self):
        """Test that caching works correctly in parallel mode."""
        model = DummyModel()
        model_manager = DummyModelManager(model)
        n_atoms = 2
        scheduler = HubScheduler(model_manager)

        # Two atoms with same initial positions
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        atom1 = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H"],
            positions=pos.copy(),
        )
        atom1.calc = ProxyCalculator(n_atoms=n_atoms)
        atom1._parallel_mode = True

        atom2 = BatchedAtoms(
            model_manager=model_manager,
            scheduler=scheduler,
            symbols=["H", "H"],
            positions=pos.copy(),
        )
        atom2.calc = ProxyCalculator(n_atoms=n_atoms)
        atom2._parallel_mode = True

        def worker1():
            atom1.get_forces()
            # Change atom1's position
            atom1.positions = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
            atom1.get_forces()

        def worker2():
            atom2.get_forces()
            # Keep atom2's position the same
            atom2.get_forces()

        g1 = greenlet(worker1)
        g2 = greenlet(worker2)
        scheduler.set_greenlet_pool([g1, g2])
        scheduler.kick_off()

        # Round 1: both compute (2 atoms batched)
        # Round 2: only atom1 needs recompute (atom2 cached)
        # So we expect 2 forward calls total
        assert model.forward_count == 2
