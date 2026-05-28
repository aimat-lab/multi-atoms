"""Hub-based greenlet scheduler for parallel MD simulations.

This module implements a trampoline (star) topology scheduler for coordinating
multiple greenlets during parallel molecular dynamics simulations.

Scheduling Topology
-------------------
The scheduler uses a hub-and-spoke pattern rather than daisy-chaining:

    Daisy-chain (BAD - O(n) stack depth):

        G1 → G2 → G3 → G4 → ...
        Each greenlet switches directly to the next, building up the call stack.

    Trampoline/Star (GOOD - O(1) stack depth):

                    ┌─────┐
              ┌────►│ Hub │◄────┐
              │     └──┬──┘     │
              │        │        │
           ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
           │ G1  │  │ G2  │  │ G3  │
           └─────┘  └─────┘  └─────┘

        All greenlets switch back to the central hub, which then dispatches
        to the next worker. Stack depth stays constant regardless of n_systems.

Execution Flow
--------------
1. Hub switches to G1, G1 runs until it calls get_forces() and yields
2. G1 switches back to Hub, Hub advances index and switches to G2
3. G2 runs until yield, switches back to Hub, Hub switches to G3
4. ... continues until all greenlets have yielded once (round complete)
5. Hub triggers batched GPU forward pass for all collected atoms
6. Hub resumes G1, which now has results and continues running
7. Process repeats until all greenlets complete

Dead Greenlet Handling
----------------------
When a greenlet finishes (dies), it is removed from the pool by swapping it
with the last element and popping. This maintains O(1) removal while allowing
the scheduler to continue with remaining greenlets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from greenlet import greenlet

if TYPE_CHECKING:
    from multiatoms.core import BatchedAtoms
    from multiatoms.model_manager import (
        ModelManager,
    )


class HubScheduler:
    """Manages greenlet switching using trampoline (star) topology.

    Instead of greenlets switching directly to each other (daisy-chain),
    all greenlets switch back to the main hub, which then decides who runs next.
    This keeps the stack depth constant at O(1) regardless of the number of systems.

    See module docstring for detailed topology diagrams and execution flow.
    """

    def __init__(self, model_manager: ModelManager):
        """Initialize RoundRobinScheduler.

        Args:
            model_manager: ModelManger to call when all greenlets have yielded
        """
        self.model_manager = model_manager
        self._greenlet_pool: list[greenlet] | None = None
        self._collected_atoms: list[BatchedAtoms] = []
        self._current_idx: int = 0
        self._main_greenlet: greenlet = greenlet.getcurrent()

    def set_greenlet_pool(self, pool: list[greenlet]) -> None:
        """Set the greenlet pool for a parallel execution round.

        Args:
            pool: List of greenlets to manage
        """
        self._greenlet_pool = pool
        self._collected_atoms = []
        self._current_idx = 0

    def yield_to_next(self, atom: BatchedAtoms) -> None:
        """Called by BatchedAtoms.get_forces() in parallel mode.

        Collects the atom and switches back to the main hub greenlet.
        The main loop in kick_off() handles advancing to the next worker
        and triggering batched forward passes when a round completes.

        Args:
            atom: The BatchedAtoms that is yielding
        """
        if self._greenlet_pool is None:
            raise ValueError("yield_to_next called while greenlet_pool is None")

        self._collected_atoms.append(atom)
        # Switch back to hub - main loop handles scheduling
        self._main_greenlet.switch()

    def kick_off(self) -> None:
        """Run the main scheduler loop using trampoline (star) topology.

        This loop acts as the central hub. Workers switch back here after yielding,
        and the hub decides who runs next. This keeps stack depth at O(1).

        The loop:
        1. Switches to the current worker greenlet
        2. When control returns (worker yielded or died), advances the index
        3. If index wraps to 0, runs the batched GPU forward pass
        4. Terminates when all greenlets are dead
        """
        if self._greenlet_pool is None:
            raise ValueError(
                "Cannot kick off pool execution while greenlet_pool is None"
            )

        while True:
            current_greenlet = self._greenlet_pool[self._current_idx]

            # Switch to worker - it will either yield back or die
            current_greenlet.switch()

            # Worker has returned control to hub. Check if it finished (dead)
            # or yielded. A dead greenlet means it completed without yielding
            # again, so no atom was collected.
            if current_greenlet.dead:
                # Remove dead greenlet: swap with last element and pop (O(1) removal)
                self._greenlet_pool[self._current_idx], self._greenlet_pool[-1] = (
                    self._greenlet_pool[-1],
                    self._greenlet_pool[self._current_idx],
                )
                self._greenlet_pool.pop()

                # All greenlets finished - exit the scheduler loop
                if not self._greenlet_pool:
                    break

                # If we removed the last element, wrap index back to start
                if self._current_idx == len(self._greenlet_pool):
                    self._current_idx = 0

                # Skip the normal index advance - we already have a new greenlet
                # at current_idx (the one we swapped in), so continue without
                # incrementing
                continue

            # Advance to next worker
            self._current_idx = (self._current_idx + 1) % len(self._greenlet_pool)

            # Round complete - run batched forward pass
            if self._current_idx == 0:
                self.model_manager.compute_energy_and_forces(self._collected_atoms)
                self._collected_atoms = []

    def clear(self) -> None:
        """Clear the greenlet pool after execution completes."""
        self._greenlet_pool = None
        self._collected_atoms = []
        self._current_idx = 0
