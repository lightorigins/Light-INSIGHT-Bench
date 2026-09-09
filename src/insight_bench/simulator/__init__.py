"""Simulator backend registry and abstract session interface.

Backends are declarative descriptors plus an abstract execution interface.
Which simulator versions exist, which one is the default, and which are
experimental are **data** (see ``SIM_BACKENDS``), so promoting a backend is a
registry change, not a code change.
"""

from insight_bench.simulator.base import (
    SIM_BACKENDS,
    SimBackendDescriptor,
    SimulatorBackend,
    SimulatorNotAvailableError,
    default_backend,
    get_backend_descriptor,
)

__all__ = [
    "SIM_BACKENDS",
    "SimBackendDescriptor",
    "SimulatorBackend",
    "SimulatorNotAvailableError",
    "default_backend",
    "get_backend_descriptor",
]
