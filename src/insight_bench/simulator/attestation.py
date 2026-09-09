"""The official-profile gate over runtime attestations.

Publication policy (ADR 0008, 2026-09): a run may enter the official
leaderboard when its attestation shows it executed on a registered
*official* backend, on Linux -- in a container or natively -- and that its
NVIDIA driver version was actually recorded. Windows runs and experimental
backends are development/debug support and are rejected for formal
publication.

Two things this gate deliberately does **not** check:

- **No driver floor.** The driver version is recorded as evidence of what the
  numbers came from; it is compared against nothing. The backend descriptor's
  ``min_driver`` is the vendor-tested minimum, kept as documentation. The
  previous L20 run on driver 535.216.03 demonstrated Isaac Sim 5.1 working
  below the vendor-listed minimum, so a floor would have rejected a run that
  did in fact execute.
- **No image digest.** The runtime image embeds NVIDIA Isaac Sim and is not
  redistributable, so a published image digest can never exist and no run
  could ever be locked to one. ``image_digest`` and ``digest_locked`` stay on
  the record as optional information.

``evaluate_official_profile`` returns the verdict plus every reason it
failed, so submission validation (e.g. the web review flow) can display and
enforce the rejection rather than silently dropping a run. A fact the
attestation does not carry -- an unrecorded driver version -- is a rejection
reason, never a skipped check.
"""

from __future__ import annotations

from insight_bench.contracts import DRIVER_VERSION_RE, RuntimeAttestation
from insight_bench.simulator.base import SimBackendDescriptor


def evaluate_official_profile(
    attestation: RuntimeAttestation,
    backend: SimBackendDescriptor,
) -> tuple[bool, tuple[str, ...]]:
    """Decide whether *attestation* qualifies for official publication.

    Pure decision logic over already-collected facts; producing a truthful
    attestation is the launcher's job. Returns ``(ok, reasons)`` where
    ``reasons`` lists every failed requirement (empty when ok).
    """
    reasons: list[str] = []
    if attestation.backend_id != backend.backend_id:
        reasons.append(
            f"attestation is for backend {attestation.backend_id!r}, not {backend.backend_id!r}"
        )
    if backend.status != "official":
        reasons.append(f"backend {backend.backend_id!r} is {backend.status}, not official")
    if attestation.os != "linux":
        reasons.append(
            "official results are accepted only from Linux runtimes; "
            f"this run used {attestation.runtime_kind} on {attestation.os}"
        )
    if not attestation.driver_version:
        reasons.append("driver version was not recorded")
    elif not DRIVER_VERSION_RE.match(attestation.driver_version):
        reasons.append(f"recorded driver version {attestation.driver_version!r} is not a version")
    return (not reasons, tuple(reasons))
