"""Unit tests for RoundRobinScheduler."""

import pytest
from greenlet import greenlet

from multiatoms.scheduler import HubScheduler


class MockModelManager:
    """Mock ModelManager that records calls."""

    def __init__(self):
        self.batch_calls: list[list] = []

    def compute_energy_and_forces(self, atoms_list):
        self.batch_calls.append(list(atoms_list))


class MockAtom:
    """Mock BatchedAtoms for testing."""

    def __init__(self, atom_id: int):
        self.atom_id = atom_id

    def __repr__(self):
        return f"MockAtom({self.atom_id})"


class TestRoundRobinSchedulerInit:
    """Tests for scheduler initialization."""

    def test_init_sets_gpu_forward(self):
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        assert scheduler.model_manager is gpu_forward
        assert scheduler._greenlet_pool is None
        assert scheduler._collected_atoms == []
        assert scheduler._current_idx == 0

    def test_set_greenlet_pool(self):
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        pool = [greenlet(lambda: None), greenlet(lambda: None)]
        scheduler.set_greenlet_pool(pool)

        assert scheduler._greenlet_pool is pool
        assert scheduler._collected_atoms == []
        assert scheduler._current_idx == 0

    def test_clear_resets_state(self):
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        pool = [greenlet(lambda: None)]
        scheduler.set_greenlet_pool(pool)
        scheduler._collected_atoms = [MockAtom(0)]
        scheduler._current_idx = 5

        scheduler.clear()

        assert scheduler._greenlet_pool is None
        assert scheduler._collected_atoms == []
        assert scheduler._current_idx == 0


class TestYieldToNext:
    """Tests for yield_to_next method."""

    def test_yield_to_next_without_pool_raises(self):
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        with pytest.raises(ValueError, match="greenlet_pool is None"):
            scheduler.yield_to_next(MockAtom(0))

    def test_yield_to_next_collects_atom(self):
        """Test that yield_to_next collects atoms and switches back to main."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        collected_in_order = []

        def worker(atom_id):
            atom = MockAtom(atom_id)
            collected_in_order.append(atom_id)
            scheduler.yield_to_next(atom)
            # After resume, record that we continued
            collected_in_order.append(f"{atom_id}_resumed")

        # Create greenlets
        g1 = greenlet(lambda: worker(1))
        g2 = greenlet(lambda: worker(2))
        scheduler.set_greenlet_pool([g1, g2])

        # Manually run the scheduler loop
        scheduler.kick_off()

        # Both workers should have yielded once, batch was triggered, then they resumed
        assert len(scheduler._collected_atoms) == 0  # cleared after batch
        assert 1 in collected_in_order
        assert 2 in collected_in_order
        assert "1_resumed" in collected_in_order
        assert "2_resumed" in collected_in_order


class TestKickOff:
    """Tests for the main scheduler loop."""

    def test_kick_off_without_pool_raises(self):
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        with pytest.raises(ValueError, match="greenlet_pool is None"):
            scheduler.kick_off()

    def test_single_greenlet_one_yield(self):
        """Single greenlet that yields once should trigger one batch."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atom = MockAtom(0)

        def worker():
            scheduler.yield_to_next(atom)

        g = greenlet(worker)
        scheduler.set_greenlet_pool([g])
        scheduler.kick_off()

        # One batch call with one atom
        assert len(gpu_forward.batch_calls) == 1
        assert len(gpu_forward.batch_calls[0]) == 1
        assert gpu_forward.batch_calls[0][0] is atom

    def test_single_greenlet_multiple_yields(self):
        """Single greenlet yielding multiple times triggers multiple batches."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atoms = [MockAtom(i) for i in range(3)]

        def worker():
            for atom in atoms:
                scheduler.yield_to_next(atom)

        g = greenlet(worker)
        scheduler.set_greenlet_pool([g])
        scheduler.kick_off()

        # Three batch calls, one per yield
        assert len(gpu_forward.batch_calls) == 3
        for i, call in enumerate(gpu_forward.batch_calls):
            assert len(call) == 1
            assert call[0] is atoms[i]

    def test_multiple_greenlets_synchronized_yields(self):
        """Multiple greenlets yielding in sync should batch together."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        n_greenlets = 4
        n_yields = 3
        atoms = [
            [MockAtom(g * 10 + y) for y in range(n_yields)] for g in range(n_greenlets)
        ]

        def make_worker(greenlet_idx):
            def worker():
                for yield_idx in range(n_yields):
                    scheduler.yield_to_next(atoms[greenlet_idx][yield_idx])

            return worker

        greenlets = [greenlet(make_worker(i)) for i in range(n_greenlets)]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # Should have n_yields batch calls, each with n_greenlets atoms
        assert len(gpu_forward.batch_calls) == n_yields
        for batch in gpu_forward.batch_calls:
            assert len(batch) == n_greenlets

    def test_greenlet_dies_early(self):
        """Test handling when one greenlet dies before others."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atom1 = MockAtom(1)
        atom2 = MockAtom(2)
        atom3 = MockAtom(3)

        execution_log = []

        def worker_short():
            execution_log.append("short_yield_1")
            scheduler.yield_to_next(atom1)
            execution_log.append("short_done")
            # Dies after one yield

        def worker_long():
            execution_log.append("long_yield_1")
            scheduler.yield_to_next(atom2)
            execution_log.append("long_yield_2")
            scheduler.yield_to_next(atom3)
            execution_log.append("long_done")

        g_short = greenlet(worker_short)
        g_long = greenlet(worker_long)
        scheduler.set_greenlet_pool([g_short, g_long])
        scheduler.kick_off()

        # First batch: both yielded (2 atoms)
        # Then short dies, long continues alone
        # Second batch: long yielded alone (1 atom)
        assert len(gpu_forward.batch_calls) == 2
        assert len(gpu_forward.batch_calls[0]) == 2  # Both in first batch
        assert len(gpu_forward.batch_calls[1]) == 1  # Only long in second

        # Verify execution order
        assert "short_done" in execution_log
        assert "long_done" in execution_log

    def test_all_greenlets_die_immediately(self):
        """Test when all greenlets die without yielding."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        def worker():
            pass  # Dies immediately

        greenlets = [greenlet(worker) for _ in range(3)]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # No batches should be called since no yields happened
        assert len(gpu_forward.batch_calls) == 0

    def test_empty_pool(self):
        """Test with empty greenlet pool raises IndexError.

        Note: This is documenting current behavior. An empty pool is not
        a valid use case for the scheduler.
        """
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        scheduler.set_greenlet_pool([])

        # Empty pool causes IndexError when trying to access first greenlet
        with pytest.raises(IndexError):
            scheduler.kick_off()

    def test_greenlets_die_in_sequence(self):
        """Test greenlets dying one by one.

        Trace through scheduler behavior:
        - Pool: [g0(1), g1(2), g2(3)] (yields remaining)
        - idx=0: g0 yields atom0, switch back, idx=1
        - idx=1: g1 yields atom1, switch back, idx=2
        - idx=2: g2 yields atom2, switch back, idx=0 -> BATCH [0,1,2]
        - idx=0: g0 dies (no yield), swap with last -> pool=[g2,g1], continue at idx=0
        - idx=0: g2 yields atom2, switch back, idx=1
        - idx=1: g1 yields atom1, switch back, idx=0 -> BATCH [2,1]
        - idx=0: g2 yields atom2, switch back, idx=1
        - idx=1: g1 dies, swap with last -> pool=[g2], continue at idx=1
        - idx=1 == len(pool)=1, so idx wraps to 0
        - idx=0: g2 dies, pool becomes empty -> break
        """
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atoms = [MockAtom(i) for i in range(3)]

        def make_worker(idx, n_yields):
            def worker():
                for _ in range(n_yields):
                    scheduler.yield_to_next(atoms[idx])

            return worker

        # g0 yields 1 time, g1 yields 2 times, g2 yields 3 times
        greenlets = [
            greenlet(make_worker(0, 1)),
            greenlet(make_worker(1, 2)),
            greenlet(make_worker(2, 3)),
        ]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # Based on actual scheduler behavior:
        # - Round 1: all 3 yield -> batch of 3
        # - g0 dies, gets swapped out
        # - Round 2: g2 and g1 yield -> batch of 2
        # - g2 yields again but then g1 dies before round completes, so g2's
        #   third yield doesn't trigger a batch before the pool empties
        assert len(gpu_forward.batch_calls) == 2
        assert len(gpu_forward.batch_calls[0]) == 3
        assert len(gpu_forward.batch_calls[1]) == 2


class TestSchedulerEdgeCases:
    """Edge case tests for scheduler behavior."""

    def test_last_greenlet_dies_wrap_around(self):
        """Test when the last greenlet in pool dies and index needs to wrap.

        Trace through:
        - Pool: [g0(2), g1(2), g2(1)]
        - idx=0: g0 yields, idx=1
        - idx=1: g1 yields, idx=2
        - idx=2: g2 yields, idx=0 -> BATCH [0,1,2]
        - idx=0: g0 yields, idx=1
        - idx=1: g1 yields, idx=2
        - idx=2: g2 dies, swap with last (itself), pop -> pool=[g0,g1]
        - idx=2 == len(pool)=2, wrap to idx=0, continue (skip batch)
        - idx=0: g0 dies, swap with g1 -> pool=[g1], continue
        - idx=0: g1 dies -> pool empty, break

        Only the first round triggers a batch due to continue statements.
        """
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atoms = [MockAtom(i) for i in range(3)]

        # g0 yields twice, g1 yields twice, g2 yields once (dies early)
        def make_worker(idx, n_yields):
            def worker():
                for _ in range(n_yields):
                    scheduler.yield_to_next(atoms[idx])

            return worker

        greenlets = [
            greenlet(make_worker(0, 2)),
            greenlet(make_worker(1, 2)),
            greenlet(make_worker(2, 1)),  # Dies after first round
        ]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # Only the first round's batch is triggered; subsequent rounds are
        # interrupted by greenlet deaths which use continue, skipping batches
        assert len(gpu_forward.batch_calls) == 1
        assert len(gpu_forward.batch_calls[0]) == 3

    def test_first_greenlet_dies_swap_behavior(self):
        """Test swap-with-last behavior when first greenlet dies."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        execution_order = []

        def make_worker(name, n_yields):
            def worker():
                for i in range(n_yields):
                    execution_order.append(f"{name}_yield_{i}")
                    scheduler.yield_to_next(MockAtom(0))
                execution_order.append(f"{name}_done")

            return worker

        # g0 dies first, g1 and g2 continue
        greenlets = [
            greenlet(make_worker("g0", 1)),
            greenlet(make_worker("g1", 2)),
            greenlet(make_worker("g2", 2)),
        ]
        scheduler.set_greenlet_pool(greenlets)
        scheduler.kick_off()

        # All workers should complete
        assert "g0_done" in execution_order
        assert "g1_done" in execution_order
        assert "g2_done" in execution_order

    def test_scheduler_reuse(self):
        """Test that scheduler can be reused after clear()."""
        gpu_forward = MockModelManager()
        scheduler = HubScheduler(gpu_forward)

        atom = MockAtom(0)

        def worker():
            scheduler.yield_to_next(atom)

        # First run
        scheduler.set_greenlet_pool([greenlet(worker)])
        scheduler.kick_off()
        scheduler.clear()

        # Second run
        scheduler.set_greenlet_pool([greenlet(worker)])
        scheduler.kick_off()
        scheduler.clear()

        # Should have two batch calls total
        assert len(gpu_forward.batch_calls) == 2
