"""Simulator backend descriptors and the abstract execution interface.

Version policy (product decision, 2026-08):

- ``isaac-5.1`` (Isaac Sim 5.1.0 / Isaac Lab 2.3) is the **default, official**
  runtime. Leaderboard-comparable numbers come from this backend only.
- ``isaac-6.0`` (Isaac Sim 6.0.1 / Isaac Lab v3.0.0-beta2.patch1) is
  supported but **experimental**: its results must never be merged or ranked
  against ``isaac-5.1`` results until it passes the published conformance
  protocol. Promotion flips ``status`` here and in the registry manifest —
  a data change, not a code change.

``image_digest`` is ``None`` on every line and gates nothing (ADR 0008): the
runtime image embeds NVIDIA Isaac Sim and is not redistributable, so no
published digest can exist. ``min_driver`` is the vendor-tested minimum,
recorded for documentation and never enforced. An execution path that cannot
run raises ``SimulatorNotAvailableError`` instead of pretending to.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from insight_bench.simulator.observation import Observation, Quat
    from insight_bench.vln_runtime.motion.pose import CameraPose

BackendStatus = Literal["official", "experimental"]


class SimulatorNotAvailableError(RuntimeError):
    """Raised when a simulator backend cannot actually execute episodes."""


@dataclass(frozen=True)
class SimBackendDescriptor:
    """Declarative facts about one simulator line. Pure data, no behavior."""

    backend_id: str
    status: BackendStatus
    isaac_sim: str
    isaac_lab: str
    python_floor: str
    # SHA-256 digest of a published runtime image, when one exists. Informational
    # only (ADR 0008): the image embeds NVIDIA Isaac Sim and is not
    # redistributable, so this is None and no gate reads it.
    image_digest: str | None
    # Tested/recommended minimum NVIDIA driver for this line, from the vendor
    # requirements page. Documentation: the attestation records the driver that
    # actually ran; nothing compares it against this value (ADR 0008).
    min_driver: str = ""
    default: bool = False
    # Audited runtime stack facts for this line (informational data).
    runtime_facts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "status": self.status,
            "isaac_sim": self.isaac_sim,
            "isaac_lab": self.isaac_lab,
            "python_floor": self.python_floor,
            "image_digest": self.image_digest,
            "min_driver": self.min_driver,
            "default": self.default,
            "runtime_facts": dict(self.runtime_facts),
        }


SIM_BACKENDS: tuple[SimBackendDescriptor, ...] = (
    SimBackendDescriptor(
        backend_id="isaac-5.1",
        status="official",
        isaac_sim="5.1.0",
        isaac_lab="2.3.0",
        python_floor="3.10",
        image_digest=None,
        min_driver="580.65.06",
        default=True,
    ),
    SimBackendDescriptor(
        backend_id="isaac-6.0",
        status="experimental",
        isaac_sim="6.0.1",
        isaac_lab="v3.0.0-beta2.patch1",
        python_floor="3.12",
        image_digest=None,
        min_driver="595.58.03",
        # Audited 6.0.1 stack (2026-08): recorded so conformance runs can
        # verify the environment they actually executed on.
        runtime_facts={
            "python": "3.12",
            "torch": "2.10 (cu128)",
            "numpy": "2.3.1",
            "driver": "595.58.03",
        },
    ),
)


def default_backend() -> SimBackendDescriptor:
    for descriptor in SIM_BACKENDS:
        if descriptor.default:
            return descriptor
    raise AssertionError("SIM_BACKENDS declares no default backend")


def get_backend_descriptor(backend_id: str) -> SimBackendDescriptor:
    for descriptor in SIM_BACKENDS:
        if descriptor.backend_id == backend_id:
            return descriptor
    known = ", ".join(item.backend_id for item in SIM_BACKENDS)
    raise KeyError(f"unknown simulator backend {backend_id!r}; known: {known}")


class SimulatorBackend(abc.ABC):
    """Abstract episode-execution interface one simulator line implements.

    Implementations live behind the ``isaac`` extra and are installed only in
    the runtime image; this module must stay importable everywhere.
    """

    @property
    @abc.abstractmethod
    def descriptor(self) -> SimBackendDescriptor:
        """Return the declarative facts for this backend."""

    @abc.abstractmethod
    def probe(self) -> dict[str, Any]:
        """Report truthfully whether this backend can execute here, and why."""

    @abc.abstractmethod
    def run_episode(self, episode: Any, policy_url: str) -> dict[str, Any]:
        """Execute one episode against a policy server and return raw outputs."""

    @abc.abstractmethod
    def capture_observation(self) -> Observation:
        """Return the current camera observation in the canonical numpy contract.

        Backends convert line-native arrays (torch tensors on 5.1, warp
        arrays on 6.0) via ``insight_bench.simulator.observation`` before
        anything above this interface sees them.
        """

    @abc.abstractmethod
    def read_camera_pose(self) -> tuple[CameraPose, Quat]:
        """Read back the current pose and canonical WXYZ orientation.

        Isaac Lab 3.x is XYZW-native; backends must convert explicitly
        (``quat_xyzw_to_wxyz``) and never leak line-native order. Gate G2
        asserts |commanded - read back| yaw < ``G2_YAW_TOLERANCE_RAD``.
        """
