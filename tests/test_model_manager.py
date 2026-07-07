"""Unit tests for ModelManager."""

from typing import List

import numpy as np
import torch
from torch import Tensor

from multiatoms.model_manager import (
    ModelManager,
)
from multiatoms.proxy_calculator import (
    ProxyCalculator,
)


class MockModel:
    """Mock ML model for testing."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self.call_count = 0
        self.last_pos = None
        self.last_batch_idx = None
        self.last_partial_charges = None

    @property
    def device(self):
        return self._device

    def __call__(self, pos, batch_idx, partial_charges=None):
        self.call_count += 1
        self.last_pos = pos
        self.last_batch_idx = batch_idx
        self.last_partial_charges = partial_charges

        # Return sum of positions as energy (one per batch)
        n_batches = batch_idx.max().item() + 1 if len(batch_idx) > 0 else 0
        energies = []
        for b in range(n_batches):
            mask = batch_idx == b
            energies.append(pos[mask].sum())
        return torch.stack(energies) if energies else torch.tensor([])

    def get_forces(self, energy, pos):
        # Return negative positions as forces (simple test pattern)
        return -pos

    def clean_up(self):
        pass


class MockBatchedAtoms:
    """Mock BatchedAtoms for testing ModelManager."""

    def __init__(
        self,
        positions: np.ndarray,
        atom_id: int = 0,
        partial_charges: np.ndarray | None = None,
    ):
        self.positions = positions
        self.atom_id = atom_id
        self._cached_positions = None
        self.calc = ProxyCalculator(n_atoms=positions.shape[0])
        self.arrays = {}
        if partial_charges is not None:
            self.arrays["partial_charges"] = partial_charges

    def __len__(self):
        return len(self.positions)


class MockModelManager(ModelManager):
    """Mock ModelManager for testing (vacuum-like, no partial charges)."""

    def __init__(self, model, device: str = "cpu"):
        super().__init__(model, device)

    def curate_batch(self, atoms_list: List["MockBatchedAtoms"]) -> dict[str, Tensor]:
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


class MockModelManagerWithCharges(ModelManager):
    """Mock ModelManager for testing with partial charges."""

    def __init__(self, model, device: str = "cpu"):
        super().__init__(model, device)

    def curate_batch(self, atoms_list: List["MockBatchedAtoms"]) -> dict[str, Tensor]:
        n_batches = len(atoms_list)
        n_atoms = len(atoms_list[0])

        positions_np = np.stack([a.positions for a in atoms_list])
        positions_np = positions_np.reshape(-1, 3) * 0.1  # A -> nm
        pos = torch.tensor(positions_np, dtype=torch.float32, device=self.device)

        batch_idx = torch.arange(n_batches, device=self.device).repeat_interleave(
            n_atoms
        )

        # Include partial charges
        charges = atoms_list[0].arrays["partial_charges"]
        charges_tensor = torch.tensor(charges, dtype=torch.float32, device=self.device)

        return {
            "pos": pos,
            "batch_idx": batch_idx,
            "partial_charges": charges_tensor.repeat(n_batches),
        }


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_init_stores_references(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        assert model_manager.model is model
        assert model_manager.device == "cpu"


class TestCurateBatch:
    """Tests for batch curation via ModelManager."""

    def test_single_atom_batch(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        atom = MockBatchedAtoms(positions)

        batched_input = model_manager.curate_batch([atom])

        # Check positions tensor shape and values
        pos_tensor = batched_input["pos"]
        assert pos_tensor.shape == (3, 3)
        # Positions should be scaled by 0.1 (Å to nm)
        expected_pos = positions * 0.1
        np.testing.assert_allclose(pos_tensor.detach().cpu().numpy(), expected_pos)

        # Check batch_idx
        assert batched_input["batch_idx"].shape == (3,)
        assert torch.all(batched_input["batch_idx"] == 0)

        # No partial charges for vacuum-like model
        assert "partial_charges" not in batched_input

    def test_multiple_atoms_batch(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        positions2 = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

        atoms = [MockBatchedAtoms(positions1), MockBatchedAtoms(positions2)]

        batched_input = model_manager.curate_batch(atoms)

        # Shape should be (n_batches * n_atoms, 3) = (4, 3)
        pos_tensor = batched_input["pos"]
        assert pos_tensor.shape == (4, 3)

        # Batch indices: [0, 0, 1, 1]
        expected_batch_idx = torch.tensor([0, 0, 1, 1])
        assert torch.all(batched_input["batch_idx"] == expected_batch_idx)

    def test_partial_charges_non_vacuum(self):
        model = MockModel()
        model_manager = MockModelManagerWithCharges(model)

        charges = np.array([0.5, -0.5])
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        atom1 = MockBatchedAtoms(positions, partial_charges=charges)
        atom2 = MockBatchedAtoms(positions, partial_charges=charges)

        batched_input = model_manager.curate_batch([atom1, atom2])

        # Partial charges should be repeated for each batch
        assert "partial_charges" in batched_input
        expected_charges = torch.tensor([0.5, -0.5, 0.5, -0.5])
        torch.testing.assert_close(batched_input["partial_charges"], expected_charges)

    def test_batch_idx_correct_for_many_atoms(self):
        model = MockModel()
        model_manager = MockModelManager(model)
        n_atoms = 5
        n_batches = 3

        atoms = [
            MockBatchedAtoms(np.random.randn(n_atoms, 3)) for _ in range(n_batches)
        ]
        batched_input = model_manager.curate_batch(atoms)

        # Check batch_idx pattern: [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
        expected = torch.arange(n_batches).repeat_interleave(n_atoms)
        assert torch.all(batched_input["batch_idx"] == expected)


class TestDistributeResults:
    """Tests for distribute_results method."""

    def test_single_atom_distribution(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        atom = MockBatchedAtoms(positions)

        forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        energy = np.array([1.5])

        model_manager.distribute_results([atom], forces, energy)

        # Check that ProxyCalculator received correct results
        calc_forces = atom.calc.get_forces()
        calc_energy = atom.calc.get_potential_energy()

        np.testing.assert_array_equal(calc_forces, forces)
        assert calc_energy == 1.5

    def test_multiple_atoms_distribution(self):
        model = MockModel()
        model_manager = MockModelManager(model)
        n_atoms = 2

        atom1 = MockBatchedAtoms(np.zeros((n_atoms, 3)))
        atom2 = MockBatchedAtoms(np.zeros((n_atoms, 3)))
        atom3 = MockBatchedAtoms(np.zeros((n_atoms, 3)))

        # Forces: (3 * 2, 3) = (6, 3)
        forces = np.array(
            [
                [1.0, 1.1, 1.2],  # atom1, particle 0
                [1.3, 1.4, 1.5],  # atom1, particle 1
                [2.0, 2.1, 2.2],  # atom2, particle 0
                [2.3, 2.4, 2.5],  # atom2, particle 1
                [3.0, 3.1, 3.2],  # atom3, particle 0
                [3.3, 3.4, 3.5],  # atom3, particle 1
            ]
        )
        energy = np.array([10.0, 20.0, 30.0])

        model_manager.distribute_results([atom1, atom2, atom3], forces, energy)

        # Verify each atom got its slice
        np.testing.assert_array_equal(atom1.calc.get_forces(), forces[0:2])
        np.testing.assert_array_equal(atom2.calc.get_forces(), forces[2:4])
        np.testing.assert_array_equal(atom3.calc.get_forces(), forces[4:6])

        assert atom1.calc.get_potential_energy() == 10.0
        assert atom2.calc.get_potential_energy() == 20.0
        assert atom3.calc.get_potential_energy() == 30.0


class TestComputeEnergyAndForces:
    """Tests for compute_energy_and_forces method."""

    def test_compute_calls_model(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom = MockBatchedAtoms(positions)

        model_manager.compute_energy_and_forces([atom])

        assert model.call_count == 1

    def test_compute_caches_positions(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom = MockBatchedAtoms(positions)

        model_manager.compute_energy_and_forces([atom])

        # Cached positions should be set
        np.testing.assert_array_equal(atom._cached_positions, positions)

    def test_compute_skips_unchanged_positions(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom = MockBatchedAtoms(positions)

        model_manager.compute_energy_and_forces([atom])
        assert model.call_count == 1

        # Run again with same positions
        model_manager.compute_energy_and_forces([atom])
        assert model.call_count == 1  # Should not call model again

    def test_compute_recomputes_on_position_change(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom = MockBatchedAtoms(positions)

        model_manager.compute_energy_and_forces([atom])
        assert model.call_count == 1

        # Change positions
        atom.positions = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        model_manager.compute_energy_and_forces([atom])
        assert model.call_count == 2

    def test_compute_mixed_cache_hit_miss(self):
        """Test batch where some atoms have changed, some haven't."""
        model = MockModel()
        model_manager = MockModelManager(model)

        pos1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pos2 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        atom1 = MockBatchedAtoms(pos1)
        atom2 = MockBatchedAtoms(pos2)

        # First run: both computed
        model_manager.compute_energy_and_forces([atom1, atom2])
        assert model.call_count == 1

        # Change only atom2's positions
        atom2.positions = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

        # Second run: only atom2 should be recomputed
        model_manager.compute_energy_and_forces([atom1, atom2])
        assert model.call_count == 2

        # Verify model was called with only 1 batch (atom2)
        assert model.last_batch_idx.max().item() == 0  # Only one batch

    def test_compute_empty_list(self):
        model = MockModel()
        model_manager = MockModelManager(model)

        model_manager.compute_energy_and_forces([])
        assert model.call_count == 0

    def test_compute_all_cached(self):
        """Test when all atoms have cached positions (nothing to compute)."""
        model = MockModel()
        model_manager = MockModelManager(model)

        positions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom = MockBatchedAtoms(positions)
        atom._cached_positions = positions.copy()  # Pre-cache

        model_manager.compute_energy_and_forces([atom])
        assert model.call_count == 0
