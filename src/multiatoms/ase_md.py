"""Optional ASE molecular-dynamics helpers for parallel simulations.

Running many ASE integrators at once (the multiatoms use case) trips a
file-descriptor leak: ASE's ``MolecularDynamics`` opens ``/dev/null`` whenever
no logfile is given, so N parallel integrators hold N descriptors open and can
exhaust the process limit. ``NullLogger`` is a no-op stand-in to pass as
``logfile`` instead, and ``SmartLangevin`` wires it in automatically.

This module is intentionally *not* imported by the package core — it is an
opt-in convenience for ASE-driven workflows. Import it explicitly:

    from multiatoms.ase_md import NullLogger, SmartLangevin
"""

from pathlib import Path
from typing import IO, Optional, Union

from ase.md.langevin import Langevin


class Singleton(type):
    """Metaclass that returns a single shared instance per class."""

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class NullLogger(metaclass=Singleton):
    """Pseudo logger that discards all input.

    Pass this to an ASE integrator/optimizer's ``logfile`` instead of ``None``:
    the default behavior, while logging nothing, opens a ``/dev/null`` file
    descriptor and causes fd inflation when many simulations run at once.
    """

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class SmartLangevin(Langevin):
    """Langevin integrator that never opens a real file for logging."""

    def openfile(
        self, file: Optional[Union[IO, str, Path]] = None, comm=None, mode="w"
    ):
        """Return ``NullLogger`` when no file is requested.

        ASE's ``MolecularDynamics`` forces ``file=None`` internally, which would
        otherwise open ``/dev/null``. A real filename is handled normally.
        """
        if file is None:
            return NullLogger()
        return super().openfile(file, comm, mode)
