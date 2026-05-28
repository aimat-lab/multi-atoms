"""Tests for the optional ASE-MD helpers."""

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

from multiatoms.ase_md import NullLogger, SmartLangevin


def _make_dyn():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = EMT()
    return SmartLangevin(
        atoms,
        timestep=1 * units.fs,
        temperature_K=300,
        friction=0.01,
        logfile=None,
    )


def test_null_logger_is_singleton():
    assert NullLogger() is NullLogger()


def test_null_logger_methods_are_noops():
    logger = NullLogger()
    # Should accept anything and never raise.
    logger.write("anything", extra=1)
    logger.flush()
    logger.close()


def test_smart_langevin_is_langevin_subclass():
    assert issubclass(SmartLangevin, Langevin)


def test_smart_langevin_openfile_none_returns_null_logger():
    dyn = _make_dyn()
    assert isinstance(dyn.openfile(None), NullLogger)


def test_smart_langevin_runs():
    dyn = _make_dyn()
    dyn.run(3)
    assert np.all(np.isfinite(dyn.atoms.get_positions()))
