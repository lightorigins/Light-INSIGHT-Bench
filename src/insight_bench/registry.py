"""Versioned registry loading, linting, and reproducible lock generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError

from insight_bench._json import load_json, sha256_file, write_stable_json
from insight_bench._paths import require_regular_file, safe_join
from insight_bench.contracts import (
    BenchmarkManifest,
    LockedResource,
    RegistryContract,
    RegistryEntry,
    RegistryLock,
    RegistryLockEntry,
)


def is_internal_fixture(manifest: BenchmarkManifest) -> bool:
    """Whether *manifest* is an internal fixture rather than an offered benchmark.

    Read off the manifest's own metadata: a fixture says so about itself, so the
    catalog does not need to know any particular slug. Both flags are honoured
    because they mean slightly different things -- ``fixture`` is "this exists to
    exercise the code", ``internal`` is "this is not for outside consumption" --
    and a manifest carrying either is not something to publish.
    """
    metadata = manifest.metadata or {}
    return bool(metadata.get("fixture")) or bool(metadata.get("internal"))


class RegistryError(ValueError):
    """Raised when registry integrity or validation fails."""


def builtin_registry_root() -> Path:
    return Path(str(files("insight_bench").joinpath("registry")))


@dataclass(frozen=True)
class Registry:
    root: Path
    contract: RegistryContract
    manifests: dict[str, BenchmarkManifest]
    entries: dict[str, RegistryEntry]

    @classmethod
    def load(cls, root: Path | None = None) -> Registry:
        registry_root = (root or builtin_registry_root()).resolve()
        index_path = registry_root / "index.json"
        try:
            require_regular_file(index_path)
            contract = RegistryContract.model_validate(load_json(index_path))
            manifests: dict[str, BenchmarkManifest] = {}
            entries: dict[str, RegistryEntry] = {}
            for entry in contract.entries:
                manifest_path = safe_join(registry_root, entry.manifest_path)
                require_regular_file(manifest_path)
                actual_hash = sha256_file(manifest_path)
                if actual_hash != entry.sha256:
                    raise RegistryError(
                        f"manifest digest mismatch for {entry.coordinate}: "
                        f"expected {entry.sha256}, got {actual_hash}"
                    )
                manifest = BenchmarkManifest.model_validate(load_json(manifest_path))
                if manifest.coordinate != entry.coordinate:
                    raise RegistryError(
                        f"registry entry {entry.coordinate} points to {manifest.coordinate}"
                    )
                cls._verify_dataset(manifest_path.parent, manifest)
                manifests[entry.coordinate] = manifest
                entries[entry.coordinate] = entry
            return cls(registry_root, contract, manifests, entries)
        except (OSError, ValidationError, ValueError) as exc:
            if isinstance(exc, RegistryError):
                raise
            raise RegistryError(str(exc)) from exc

    @staticmethod
    def _verify_dataset(manifest_dir: Path, manifest: BenchmarkManifest) -> None:
        dataset = manifest.dataset
        if dataset.availability != "bundled":
            return
        assert dataset.path is not None
        assert dataset.sha256 is not None
        dataset_path = safe_join(manifest_dir, dataset.path)
        require_regular_file(dataset_path)
        actual_hash = sha256_file(dataset_path)
        if actual_hash != dataset.sha256:
            raise RegistryError(
                f"dataset digest mismatch for {manifest.coordinate}: "
                f"expected {dataset.sha256}, got {actual_hash}"
            )

    def catalog(self, *, include_fixtures: bool = True) -> tuple[BenchmarkManifest, ...]:
        """Every registered manifest, or only the publishable ones.

        ``include_fixtures=False`` drops internal fixtures. The predicate is
        :func:`is_internal_fixture` -- a property the manifest declares about
        itself -- not a list of slugs. A slug denylist has to be edited every time
        a fixture is added, and the fixture is public until someone remembers.
        """
        manifests = tuple(self.manifests[key] for key in sorted(self.manifests))
        if include_fixtures:
            return manifests
        return tuple(m for m in manifests if not is_internal_fixture(m))

    def public_catalog(self) -> tuple[BenchmarkManifest, ...]:
        """The benchmarks this distribution offers. Fixtures are not among them."""
        return self.catalog(include_fixtures=False)

    def get(self, reference: str) -> BenchmarkManifest:
        if "@" in reference:
            coordinate = reference
        else:
            candidates = [
                manifest
                for manifest in self.manifests.values()
                if manifest.benchmark_id == reference
            ]
            if not candidates:
                raise RegistryError(f"unknown benchmark {reference!r}")
            coordinate = max(candidates, key=lambda item: _semver_key(item.version)).coordinate
        try:
            return self.manifests[coordinate]
        except KeyError as exc:
            available = ", ".join(sorted(self.manifests))
            raise RegistryError(
                f"unknown benchmark {coordinate!r}; available: {available}"
            ) from exc

    def manifest_path(self, manifest: BenchmarkManifest) -> Path:
        entry = self.entries[manifest.coordinate]
        return safe_join(self.root, entry.manifest_path)

    def dataset_path(self, manifest: BenchmarkManifest) -> Path:
        if manifest.dataset.availability != "bundled" or manifest.dataset.path is None:
            raise RegistryError(f"{manifest.coordinate} has no bundled dataset")
        return safe_join(self.manifest_path(manifest).parent, manifest.dataset.path)


def lint_registry(root: Path | None = None) -> tuple[str, ...]:
    """Return validation issues; an empty tuple means the registry is valid."""
    try:
        Registry.load(root)
    except RegistryError as exc:
        return (str(exc),)
    return ()


def lock_registry(root: Path | None = None, output: Path | None = None) -> RegistryLock:
    registry = Registry.load(root)
    lock_entries: list[RegistryLockEntry] = []
    for manifest in registry.catalog():
        entry = registry.entries[manifest.coordinate]
        resources: list[LockedResource] = []
        if manifest.dataset.availability == "bundled":
            dataset_path = registry.dataset_path(manifest)
            resources.append(
                LockedResource(
                    path=dataset_path.relative_to(registry.root).as_posix(),
                    sha256=sha256_file(dataset_path),
                )
            )
        lock_entries.append(
            RegistryLockEntry(
                coordinate=manifest.coordinate,
                manifest_path=entry.manifest_path,
                manifest_sha256=entry.sha256,
                resources=tuple(resources),
            )
        )
    lock = RegistryLock(
        registry_version=registry.contract.registry_version,
        registry_sha256=sha256_file(registry.root / "index.json"),
        entries=tuple(lock_entries),
    )
    if output is not None:
        write_stable_json(output, lock)
    return lock


def registry_coordinates(manifests: Iterable[BenchmarkManifest]) -> tuple[str, ...]:
    return tuple(sorted(manifest.coordinate for manifest in manifests))


def _semver_key(value: str) -> tuple[int, int, int]:
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)
