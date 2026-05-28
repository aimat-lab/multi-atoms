import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class ProxyCalculator(Calculator):
    """ASE Calculator that receives references to result arrays from ModelManager.

    Slices into shared force/energy arrays on access (zero-copy for single atom,
    slice view for batched atoms).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, n_atoms: int, **kwargs):
        """Initialize ProxyCalculator.

        Args:
            **kwargs: Additional arguments for Calculator base class
        """
        super().__init__(**kwargs)
        self._forces: np.ndarray | None = None
        self._energy: np.ndarray | None = None
        self._atom_index: int = 0
        self._n_atoms: int = n_atoms

    def set_results(
        self, forces: np.ndarray, energy: np.ndarray, atom_index: int
    ) -> None:
        """Store references to result arrays from ModelManager.

        Args:
            forces: Reference to the full forces array (n_systems * n_atoms, 3)
            energy: Reference to the full energy array (n_systems,)
            atom_index: Index of this atom in the batch
            n_atoms: Number of atoms per system
        """
        self._forces = forces
        self._energy = energy
        self._atom_index = atom_index

    def get_forces(self, atoms=None, **kwargs) -> np.ndarray:
        """Return this atom's slice of the forces array."""
        if self._forces is None:
            raise RuntimeError("Forces not set. Call set_results first.")
        start = self._atom_index * self._n_atoms
        return self._forces[start : start + self._n_atoms]

    def get_potential_energy(self, atoms=None, **kwargs) -> float:
        """Return this atom's energy.

        Args:
            atoms: Ignored (for ASE interface compatibility)
            **kwargs: Accepts force_consistent and other ASE kwargs (ignored)
        """
        if self._energy is None:
            raise RuntimeError("Energy not set. Call set_results first.")
        return float(self._energy[self._atom_index])

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """Calculate energy and forces for given atoms.

        This method is called by ASE's get_forces() and get_potential_energy().
        We populate self.results from our stored references.
        """
        if self._forces is None or self._energy is None:
            raise RuntimeError(
                "Results not set. ModelManager must call set_results first."
            )

        self.results["energy"] = self.get_potential_energy(atoms)
        self.results["forces"] = self.get_forces(atoms)
