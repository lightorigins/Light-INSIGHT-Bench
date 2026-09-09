"""Version 1 public Pydantic contracts.

The corresponding JSON Schemas live in ``insight_bench/schemas/v1``. Additive
changes remain within v1; incompatible changes require a new versioned module
and schema directory.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:  # Python 3.10: the Isaac Sim 5.1 runtime interpreter floor.
    from typing_extensions import Self

from insight_bench._paths import validate_relative_path

CONTRACT_VERSION: Literal["1.0.0"] = "1.0.0"
_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
DRIVER_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
"""A dotted-numeric NVIDIA driver version (e.g. ``535.216.03``).

The one rule for what counts as a *recorded* driver, applied by both the
``RuntimeAttestation`` validator and the official-profile gate
(``insight_bench.simulator.attestation``), so the two cannot disagree.
"""
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractModel(BaseModel):
    """Strict, immutable base for wire contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def _check_slug(value: str) -> str:
    if not _SLUG.fullmatch(value):
        raise ValueError("must be a lowercase portable identifier")
    return value


def _check_semver(value: str) -> str:
    if not _SEMVER.fullmatch(value):
        raise ValueError("must be a three-part semantic version")
    return value


def _check_sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError("must be a lowercase SHA-256 hex digest")
    return value


class LicenseDeclaration(ContractModel):
    status: Literal["declared", "placeholder"]
    spdx_id: str | None = None
    notice: str

    @model_validator(mode="after")
    def declared_license_has_spdx_id(self) -> Self:
        if self.status == "declared" and not self.spdx_id:
            raise ValueError("a declared license requires spdx_id")
        if self.status == "placeholder" and self.spdx_id is not None:
            raise ValueError("a placeholder license cannot claim an SPDX identifier")
        return self


class DatasetResource(ContractModel):
    availability: Literal["bundled", "user-provided", "unconfigured"]
    format: Literal["json"] = "json"
    path: str | None = None
    sha256: str | None = None
    notice: str = ""

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str | None) -> str | None:
        return None if value is None else validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str | None) -> str | None:
        return None if value is None else _check_sha256(value)

    @model_validator(mode="after")
    def bundled_resource_is_pinned(self) -> Self:
        if self.availability == "bundled" and (self.path is None or self.sha256 is None):
            raise ValueError("a bundled dataset requires path and sha256")
        if self.availability == "unconfigured" and (
            self.path is not None or self.sha256 is not None
        ):
            raise ValueError("an unconfigured dataset cannot contain a path or digest")
        return self


class TaskDefinition(ContractModel):
    task_id: str
    description: str
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()

    _task_id = field_validator("task_id")(_check_slug)


class RuntimeSpec(ContractModel):
    runner: str
    model_execution: Literal["local-only"] = "local-only"
    network_access: Literal["disabled"] = "disabled"

    _runner = field_validator("runner")(_check_slug)


class AssetLicense(ContractModel):
    """Provenance and licence terms for one user-provided scene-asset family.

    Part of the published benchmark contract: whether an outside reader can
    reproduce a number at all depends on these terms, so they are typed
    manifest data rather than README prose.
    """

    family: str
    tier: Literal["public", "eula_gated"]
    license: str
    obtain_url: str
    notes: str = ""

    _family = field_validator("family")(_check_slug)


class RuntimeVariant(ContractModel):
    """One simulator runtime line a benchmark can execute on.

    The manifest is the authoritative record of which lines exist, which is
    the default, and which image digests are published; the in-code backend
    descriptors must agree with it (pinned by a gate test).
    """

    backend_id: str
    status: Literal["official", "experimental"]
    isaac_sim: str
    isaac_lab: str
    image_digest: str | None = None
    default: bool = False

    _backend_id = field_validator("backend_id")(_check_slug)

    @field_validator("image_digest")
    @classmethod
    def valid_image_digest(cls, value: str | None) -> str | None:
        return None if value is None else _check_sha256(value)


class BenchmarkManifest(ContractModel):
    """Registry manifest for one immutable benchmark release."""

    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    benchmark_id: str
    version: str
    title: str
    description: str
    license: LicenseDeclaration
    runtime: RuntimeSpec
    dataset: DatasetResource
    tasks: tuple[TaskDefinition, ...] = Field(min_length=1)
    metrics: tuple[str, ...] = Field(min_length=1)
    runnable: bool
    requires_configuration: tuple[str, ...] = ()
    asset_licenses: tuple[AssetLicense, ...] = ()
    runtime_variants: tuple[RuntimeVariant, ...] = ()
    comparability_notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    _benchmark_id = field_validator("benchmark_id")(_check_slug)
    _version = field_validator("version")(_check_semver)

    @field_validator("metrics")
    @classmethod
    def valid_metrics(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _check_slug(value)
        if len(set(values)) != len(values):
            raise ValueError("metric identifiers must be unique")
        return values

    @model_validator(mode="after")
    def runnable_manifest_is_complete(self) -> Self:
        """``runnable`` means *nothing is undeclared*, not *everything is bundled*.

        The original rule read ``availability != "bundled" -> not runnable``,
        which made a published suite whose dataset the user downloads
        separately unrepresentable: the licence-gated objnav suite is a
        runnable benchmark whose episode file and scene assets this SDK must
        never ship or fetch. Conflating the two questions did not make anything
        safer -- it made the honest manifest impossible to write.

        So the check now asks what completeness actually requires, and asks
        *more* of a user-provided dataset than of a bundled one: it must pin the
        digest of the exact published file (the runner verifies the user's copy
        against it) and it must name, in ``requires_configuration``, everything
        the user has to supply. A bundled dataset keeps the original rule that
        it may require no configuration at all.
        """
        if self.runnable:
            if self.license.status != "declared":
                raise ValueError("a runnable manifest requires a declared license")
            if self.dataset.availability == "unconfigured":
                raise ValueError("a runnable manifest requires a bundled or user-provided dataset")
            if self.dataset.availability == "user-provided":
                if self.dataset.sha256 is None:
                    raise ValueError(
                        "a runnable user-provided dataset requires the sha256 of the published file"
                    )
                if not self.requires_configuration:
                    raise ValueError(
                        "a runnable user-provided dataset must declare what the user "
                        "supplies in requires_configuration"
                    )
            elif self.requires_configuration:
                raise ValueError(
                    "a runnable manifest with a bundled dataset cannot require configuration"
                )
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task identifiers must be unique")
        families = [item.family for item in self.asset_licenses]
        if len(families) != len(set(families)):
            raise ValueError("asset licence families must be unique")
        backend_ids = [variant.backend_id for variant in self.runtime_variants]
        if len(backend_ids) != len(set(backend_ids)):
            raise ValueError("runtime variant backend identifiers must be unique")
        if self.runtime_variants and sum(variant.default for variant in self.runtime_variants) != 1:
            raise ValueError("exactly one runtime variant must be the default")
        return self

    @property
    def coordinate(self) -> str:
        return f"{self.benchmark_id}@{self.version}"


class RegistryEntry(ContractModel):
    benchmark_id: str
    version: str
    manifest_path: str
    sha256: str

    _benchmark_id = field_validator("benchmark_id")(_check_slug)
    _version = field_validator("version")(_check_semver)
    _manifest_path = field_validator("manifest_path")(validate_relative_path)
    _sha256 = field_validator("sha256")(_check_sha256)

    @property
    def coordinate(self) -> str:
        return f"{self.benchmark_id}@{self.version}"


class RegistryContract(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    registry_version: str
    entries: tuple[RegistryEntry, ...]

    _registry_version = field_validator("registry_version")(_check_semver)

    @model_validator(mode="after")
    def unique_entries(self) -> Self:
        coordinates = [entry.coordinate for entry in self.entries]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("registry coordinates must be unique")
        return self


class LockedResource(ContractModel):
    path: str
    sha256: str

    _path = field_validator("path")(validate_relative_path)
    _sha256 = field_validator("sha256")(_check_sha256)


class RegistryLockEntry(ContractModel):
    coordinate: str
    manifest_path: str
    manifest_sha256: str
    resources: tuple[LockedResource, ...] = ()

    _manifest_path = field_validator("manifest_path")(validate_relative_path)
    _manifest_sha256 = field_validator("manifest_sha256")(_check_sha256)


class RegistryLock(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    registry_version: str
    registry_sha256: str
    entries: tuple[RegistryLockEntry, ...]

    _registry_version = field_validator("registry_version")(_check_semver)
    _registry_sha256 = field_validator("registry_sha256")(_check_sha256)


class AdapterDescriptor(ContractModel):
    adapter_id: str
    model_id: str
    protocol_version: Literal["1.0.0"] = CONTRACT_VERSION
    deterministic: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    _adapter_id = field_validator("adapter_id")(_check_slug)


class AdapterRequest(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    request_id: str = Field(pattern=r"^req-[0-9a-f]{16}$")
    benchmark_id: str
    benchmark_version: str
    task_id: str
    episode_id: str = Field(min_length=1, max_length=256)
    step_index: int = Field(default=0, ge=0)
    instruction: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

    _benchmark_id = field_validator("benchmark_id")(_check_slug)
    _benchmark_version = field_validator("benchmark_version")(_check_semver)
    _task_id = field_validator("task_id")(_check_slug)


class AdapterAction(ContractModel):
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    _name = field_validator("name")(_check_slug)


class AdapterResponse(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    request_id: str = Field(pattern=r"^req-[0-9a-f]{16}$")
    adapter: AdapterDescriptor
    status: Literal["success", "error"]
    output: str = ""
    actions: tuple[AdapterAction, ...] = ()
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_status(self) -> Self:
        if self.status == "success" and self.error is not None:
            raise ValueError("a successful response cannot contain an error")
        if self.status == "error" and not self.error:
            raise ValueError("an error response requires an error message")
        return self


class EpisodeRunResult(ContractModel):
    episode_id: str
    task_id: str
    status: Literal["passed", "failed", "error"]
    response: AdapterResponse | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None

    _task_id = field_validator("task_id")(_check_slug)

    @model_validator(mode="after")
    def coherent_episode_status(self) -> Self:
        if self.status == "error" and not self.error:
            raise ValueError("an errored episode requires an error message")
        if self.status != "error" and self.response is None:
            raise ValueError("a scored episode requires an adapter response")
        return self


class RuntimeAttestation(ContractModel):
    """Where and how a simulator run actually executed.

    Publication policy (ADR 0008, 2026-09): a run is leaderboard-eligible
    when it executed on an *official* backend, on Linux -- in a container or
    natively -- and its NVIDIA driver version was actually recorded, as a
    dotted-numeric version such as ``535.216.03`` (the same rule the
    official-profile gate applies; a placeholder like ``unknown`` is not a
    recorded driver). The driver version is evidence, not a floor: nothing
    compares it against a minimum. ``image_digest`` and ``digest_locked`` are optional,
    informational fields that gate nothing; the runtime image embeds NVIDIA
    Isaac Sim and cannot be redistributed, so no published digest exists for
    a run to be locked to. Windows runs and experimental backends are
    development and debugging support and must carry ``publishable: false``;
    submission validation may reject formal publication on this record. The
    in-model validator enforces the necessary conditions; the check of
    ``backend_id`` against the registry descriptor happens in the
    official-profile gate.
    """

    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    backend_id: str
    backend_status: Literal["official", "experimental"]
    launcher_id: str
    runtime_kind: Literal["docker", "native"]
    os: Literal["linux", "windows"]
    image_digest: str | None = None
    digest_locked: bool = False
    driver_version: str = ""
    # Observed Isaac stack, on the same terms as ``driver_version``: read off the
    # machine that ran, recorded as evidence, compared against nothing. A
    # backend descriptor *declares* the line's versions; these say what was
    # actually there. They differ more often than they should -- an Isaac Lab
    # install reports its package version (``0.46.3``) while its repository
    # carries a framework version (``2.2.1``), and a release-candidate Isaac Sim
    # reports ``5.1.0-rc.6`` against a declared ``5.1.0`` -- so a run that says
    # nothing here is indistinguishable from one on the declared stack, which is
    # the confusion these exist to remove. ``""`` means unread, never a guess.
    isaac_lab_version: str = ""
    isaac_sim_version: str = ""
    python_version: str = ""
    publishable: bool = False

    _backend_id = field_validator("backend_id")(_check_slug)
    _launcher_id = field_validator("launcher_id")(_check_slug)

    @field_validator("image_digest")
    @classmethod
    def valid_image_digest(cls, value: str | None) -> str | None:
        return None if value is None else _check_sha256(value)

    @model_validator(mode="after")
    def publishable_requires_the_official_profile(self) -> Self:
        if self.publishable:
            if self.backend_status != "official":
                raise ValueError("publishable results require an official backend")
            if self.os != "linux":
                raise ValueError("publishable results require a Linux runtime")
            if not self.driver_version:
                raise ValueError("publishable results require a recorded driver version")
            if not DRIVER_VERSION_RE.match(self.driver_version):
                raise ValueError(
                    "publishable results require a dotted-numeric driver version, "
                    f"not {self.driver_version!r}"
                )
        return self


class RunResult(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    run_fingerprint: str
    benchmark_id: str
    benchmark_version: str
    adapter: AdapterDescriptor
    seed: int = Field(ge=0, le=2**32 - 1)
    status: Literal["completed", "partial", "failed"]
    episodes: tuple[EpisodeRunResult, ...] = Field(min_length=1)
    metrics: dict[str, float]
    # None for local fixture runs; required by publication policy for
    # simulator runs, where it records the runtime the numbers came from.
    runtime_attestation: RuntimeAttestation | None = None

    _run_fingerprint = field_validator("run_fingerprint")(_check_sha256)
    _benchmark_id = field_validator("benchmark_id")(_check_slug)
    _benchmark_version = field_validator("benchmark_version")(_check_semver)

    @model_validator(mode="after")
    def unique_episode_ids(self) -> Self:
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode identifiers must be unique")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_runtime_attestation(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Leave ``runtime_attestation`` out of the payload when there is none.

        The field was added inside schema 1.0.0, so consumers still holding a
        pre-attestation copy of the run-result schema — which is
        ``additionalProperties: false`` — reject the key on sight. A local
        fixture run has no runtime to attest to, and emitting an explicit
        ``null`` broke those readers for no information gained. An
        attestation that exists is always serialized; only its absence is
        expressed by absence.
        """
        payload: dict[str, Any] = handler(self)
        if payload.get("runtime_attestation") is None:
            payload.pop("runtime_attestation", None)
        return payload


class EvidenceArtifact(ContractModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str
    role: Literal["run-result", "supporting"]
    sanitization: Literal["structured-redaction", "text-redaction", "binary-none"]

    _path = field_validator("path")(validate_relative_path)
    _sha256 = field_validator("sha256")(_check_sha256)


class EvidenceManifest(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    pack_id: str
    run_id: str = Field(pattern=r"^run-[0-9a-f]{16}$")
    run_fingerprint: str
    artifacts: tuple[EvidenceArtifact, ...] = Field(min_length=1)
    redactions: tuple[str, ...] = ()

    _pack_id = field_validator("pack_id")(_check_sha256)
    _run_fingerprint = field_validator("run_fingerprint")(_check_sha256)

    @model_validator(mode="after")
    def unique_artifact_paths(self) -> Self:
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")
        return self


class BenchEpisode(ContractModel):
    episode_id: str = Field(min_length=1, max_length=256)
    task_id: str
    instruction: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_output: str

    _task_id = field_validator("task_id")(_check_slug)


class EpisodeDataset(ContractModel):
    schema_version: Literal["1.0.0"] = CONTRACT_VERSION
    episodes: tuple[BenchEpisode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_episode_ids(self) -> Self:
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("episode identifiers must be unique")
        return self


PUBLIC_SCHEMAS: dict[str, type[ContractModel]] = {
    "adapter-request": AdapterRequest,
    "adapter-response": AdapterResponse,
    "benchmark-manifest": BenchmarkManifest,
    "evidence-manifest": EvidenceManifest,
    "registry": RegistryContract,
    "run-result": RunResult,
    "runtime-attestation": RuntimeAttestation,
}
