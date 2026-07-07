"""Model manager for batched model input/output processing.

This module provides the abstract base class for customizing how atomic systems
are batched for GPU computation.

Example
-------
Implementing a custom ModelManager for a graph neural network model:

    class MyGNNModelManager(ModelManager):
        '''ModelManager for a GNN-based energy/force model.'''

        def __init__(self, model: torch.nn.Module, device: str, cutoff: float = 5.0):
            super().__init__(model, device)
            self.cutoff = cutoff

        def curate_batch(self, atoms_list: List[BatchedAtoms]) -> dict[str, Tensor]:
            '''Build batched graph input from atoms.'''
            all_pos = []
            all_z = []
            batch_idx = []

            for i, atoms in enumerate(atoms_list):
                pos = torch.tensor(
                    atoms.positions, dtype=torch.float32, device=self.device
                )
                z = torch.tensor(
                    atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
                )
                all_pos.append(pos)
                all_z.append(z)
                batch_idx.append(
                    torch.full((len(atoms),), i, dtype=torch.long, device=self.device)
                )

            return {
                "pos": torch.cat(all_pos),
                "z": torch.cat(all_z),
                "batch": torch.cat(batch_idx),
            }

        def post_process_hook(
            self, forces: np.ndarray, energy: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            '''Convert from eV to kcal/mol.'''
            EV_TO_KCAL = 23.0609
            return forces * EV_TO_KCAL, energy * EV_TO_KCAL
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from multiatoms.core import BatchedAtoms


class ModelManager(ABC):
    """Abstract base class for batched model input/output processing.

    Users must implement:
        - curate_batch(): Convert list of atoms to model input tensors

    Users can optionally override:
        - post_process_hook(): Modify forces/energy before distribution (e.g., scaling)
        - run_model(): Change how the model is called (default: model + get_forces)

    The base class provides:
        - distribute_results(): Standard result distribution to ProxyCalculators

    See module docstring for a complete implementation example.
    """

    def __init__(self, model: torch.nn.Module, device: str):
        """Initialize ModelManager.

        Args:
            model: The ML model to use for predictions
            device: Device for computation (e.g., "cuda" or "cpu")
        """
        self.model = model
        self.device = device

    def compute_energy_and_forces(self, atoms_list: List["BatchedAtoms"]) -> None:
        """This is the main entry point into the batched model inference.
           Typically no adjustments needed here.

        Args:
            atoms_list: List of BatchedAtoms to process in a batch
        """
        # 1. Filter out atoms whose positions haven't changed (per-atom caching)
        atoms_to_compute: list[BatchedAtoms] = []
        for atom in atoms_list:
            if atom._cached_positions is None or not np.array_equal(
                atom._cached_positions, atom.positions
            ):
                atoms_to_compute.append(atom)

        # All atoms have unchanged positions, nothing to compute
        if not atoms_to_compute:
            return

        # 2. Run the model (curation + forward + post-process)
        forces, energy = self._infer(atoms_to_compute)

        # 3. Standard result distribution
        self.distribute_results(atoms_to_compute, forces, energy)

        # 4. Update per-atom position cache for computed atoms
        for atom in atoms_to_compute:
            atom._cached_positions = atom.positions.copy()

    def _infer(
        self, atoms_to_compute: List["BatchedAtoms"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the model on the systems that need new forces -> ``(forces, energy)``.

        Curation + forward + to-numpy + post-process, with no caching or result
        distribution. Factored out so it can be shared by the local path
        (``compute_energy_and_forces``) and the multi-process GPU force server
        (``multiatoms.poly_atoms``): the server runs this on the systems a worker
        shipped over and sends the arrays back, while ``RemoteModelManager``
        overrides it to do the shipping.
        """
        batched_input = self.curate_batch(atoms_to_compute)
        energy_raw, forces_raw = self.model_forward(batched_input)
        energy = energy_raw.flatten().cpu().detach().numpy()
        forces = forces_raw.cpu().detach().numpy()
        return self.post_process_hook(forces, energy)

    @abstractmethod
    def curate_batch(self, atoms_list: List["BatchedAtoms"]) -> dict[str, Tensor]:
        """Convert atoms list to batched model input tensors.

        Args:
            atoms_list: List of BatchedAtoms to process

        Returns:
            Dict of tensors ready for the model (e.g., {"pos": ..., "batch_idx": ...})
        """
        pass

    def model_forward(self, batched_input: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Execute model forward pass and compute forces.

        Default implementation assumes model returns energy and has get_forces method.
        Override if your model has a different interface.

        Args:
            batched_input: Output from curate_batch()

        Returns:
            Tuple of (energy_tensor, forces_tensor)
        """
        batched_input["pos"].requires_grad_(True)
        with torch.set_grad_enabled(True):
            energy = self.model(**batched_input)
            forces = self.model.get_forces(energy, batched_input["pos"])
        return energy, forces

    def post_process_hook(
        self, forces: np.ndarray, energy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optional hook to modify forces/energy before distribution.

        Override to apply scaling, unit conversion, etc.
        Default: returns inputs unchanged.

        Args:
            forces: Forces array (n_total_atoms, 3)
            energy: Energy array (n_systems,)

        Returns:
            Tuple of (processed_forces, processed_energy)
        """
        return forces, energy

    def distribute_results(
        self,
        atoms_list: List["BatchedAtoms"],
        forces: np.ndarray,
        energy: np.ndarray,
    ) -> None:
        """Distribute results to each atom's ProxyCalculator.

        This is the standard implementation - typically no need to override.

        Args:
            atoms_list: List of atoms to receive results
            forces: Full forces array (n_systems * n_atoms, 3)
            energy: Full energy array (n_systems,)
            n_atoms: Number of atoms per system
        """
        for i, atom in enumerate(atoms_list):
            atom.calc.set_results(forces, energy, atom_index=i)

    def clean_up(self) -> None:
        """Clean up model resources.

        Override if your model needs teardown (e.g., multiprocessing workers).
        Default: calls the model's ``clean_up()`` if it has one.
        """
        if hasattr(self.model, "clean_up"):
            self.model.clean_up()
