from __future__ import annotations

import hashlib

import pytest

from insight_bench.registry import (
    Registry,
    RegistryError,
    builtin_registry_root,
    lint_registry,
    lock_registry,
)

COORDINATES = ("insight-bench-v1@1.0.0",)


def test_the_registry_resolves_a_suite_and_ships_no_dataset_bytes() -> None:
    """The two ways a caller names a suite, and what the registry will not hand back.

    That the published list is exactly this one coordinate is asserted in
    ``tests/gate/test_retired_coordinates_fail_closed.py``. What is pinned here
    is the resolution the CLI depends on -- a bare ``benchmark_id`` picking the
    highest version, a full coordinate returning that exact version -- and the
    fact that no episode bytes ship behind it. The suite's episode file is
    licence-gated and user-provided, so there is no bundled path to resolve and
    asking for one is refused rather than guessed at.
    """
    registry = Registry.load()
    assert registry.get("insight-bench-v1").coordinate == "insight-bench-v1@1.0.0"
    assert registry.get("insight-bench-v1@1.0.0").version == "1.0.0"
    for manifest in registry.catalog():
        assert manifest.runnable is True
        assert manifest.dataset.availability == "user-provided"
        with pytest.raises(RegistryError, match="no bundled dataset"):
            registry.dataset_path(manifest)


def test_registry_lint_and_lock_pin_every_manifest() -> None:
    """A clean lint, and a lock that pins the bytes it claims to pin.

    The lock used to be checked through the one fixture that carried a bundled
    ``episodes.json``; with no bundled dataset left in the distribution, the
    resources list is empty by construction and the manifest digests are the
    whole of what a lock can pin. One published suite is still enough to catch
    drift: :func:`lock_registry` copies each digest out of ``index.json``, and
    the digest is recomputed here from the manifest on disk, so a manifest whose
    bytes moved away from its pin fails -- either at ``lint_registry`` and the
    ``Registry.load`` inside ``lock_registry``, which refuse a mismatched
    digest outright, or on the comparison below if that guard were ever lost.
    """
    assert lint_registry() == ()
    lock = lock_registry()
    assert len(lock.registry_sha256) == 64
    assert tuple(entry.coordinate for entry in lock.entries) == COORDINATES
    root = builtin_registry_root()
    for entry in lock.entries:
        on_disk = hashlib.sha256((root / entry.manifest_path).read_bytes()).hexdigest()
        assert entry.manifest_sha256 == on_disk
        assert entry.resources == ()
