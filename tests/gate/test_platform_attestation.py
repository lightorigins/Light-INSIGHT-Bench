"""Publication policy: runtime attestation and the official-profile gate."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from insight_bench.contracts import RunResult, RuntimeAttestation
from insight_bench.simulator.attestation import evaluate_official_profile
from insight_bench.simulator.base import SIM_BACKENDS, default_backend, get_backend_descriptor
from insight_bench.simulator.isaac import backend as isaac_backend

_DIGEST = "a" * 64


def _docker_attestation(**overrides: object) -> RuntimeAttestation:
    """A publishable Docker run as this build actually produces it: no image digest."""
    payload: dict[str, object] = {
        "backend_id": "isaac-5.1",
        "backend_status": "official",
        "launcher_id": "docker",
        "runtime_kind": "docker",
        "os": "linux",
        "image_digest": None,
        "digest_locked": False,
        "driver_version": "580.65.06",
        "publishable": True,
    }
    payload.update(overrides)
    return RuntimeAttestation.model_validate(payload)


def _native_attestation(**overrides: object) -> RuntimeAttestation:
    """A publishable native Linux run: the case ADR 0008 admits."""
    return _docker_attestation(
        **{"launcher_id": "linux-native", "runtime_kind": "native", **overrides}
    )


# --- contract-level coherence ------------------------------------------------


def test_publishable_requires_an_official_linux_backend_with_a_recorded_driver() -> None:
    # Docker or native, with or without an image digest: neither decides it.
    assert _docker_attestation().publishable is True
    assert _native_attestation().publishable is True
    assert _docker_attestation(image_digest=_DIGEST, digest_locked=True).publishable is True

    for bad in (
        {"runtime_kind": "native", "os": "windows", "launcher_id": "windows-native"},
        {"backend_status": "experimental"},
        {"driver_version": ""},
        {"driver_version": "   "},
    ):
        with pytest.raises(ValidationError):
            _docker_attestation(**bad)


def test_the_validator_applies_the_gates_dotted_numeric_driver_rule() -> None:
    # Contract and gate agree on what "recorded" means: a dotted-numeric
    # version. A placeholder can therefore never travel with `publishable:
    # true`, and the two cannot drift because they share the one regex.
    assert _docker_attestation(driver_version="535.216.03").publishable is True
    for junk in ("unknown", "535", "Driver Not Loaded", "v.x.y"):
        with pytest.raises(ValidationError, match="dotted-numeric"):
            _docker_attestation(driver_version=junk)
        # Still representable as a fact about a run that does not publish.
        assert _docker_attestation(driver_version=junk, publishable=False).publishable is False


def test_the_validator_no_longer_asks_for_a_container_or_a_digest() -> None:
    # The three conditions ADR 0008 dropped, each alone on an otherwise
    # publishable record, must all still validate.
    assert _docker_attestation(runtime_kind="native", launcher_id="linux-native").publishable
    assert _docker_attestation(digest_locked=False).publishable
    assert _docker_attestation(image_digest=None).publishable


def test_windows_runs_are_representable_but_only_as_unpublishable() -> None:
    attestation = _docker_attestation(
        runtime_kind="native", os="windows", launcher_id="windows-native", publishable=False
    )
    assert attestation.publishable is False


def test_run_result_carries_an_optional_attestation() -> None:
    fields = RunResult.model_fields
    assert "runtime_attestation" in fields
    assert fields["runtime_attestation"].default is None


# --- the official-profile gate ----------------------------------------------


def test_official_profile_accepts_docker_and_native_linux_with_a_recorded_driver() -> None:
    # The real registry state: no image digest is published on any backend.
    backend = get_backend_descriptor("isaac-5.1")
    assert backend.image_digest is None
    for attestation in (_docker_attestation(), _native_attestation()):
        ok, reasons = evaluate_official_profile(attestation, backend)
        assert ok, reasons
        assert reasons == ()


def test_official_profile_rejects_windows_and_foreign_backends_with_reasons() -> None:
    backend = get_backend_descriptor("isaac-5.1")

    windows = _docker_attestation(
        runtime_kind="native", os="windows", launcher_id="windows-native", publishable=False
    )
    ok, reasons = evaluate_official_profile(windows, backend)
    assert not ok and any("only from Linux" in reason for reason in reasons)

    # The validator cannot know the registry; the gate does.
    foreign = _docker_attestation(backend_id="isaac-6.0")
    ok, reasons = evaluate_official_profile(foreign, backend)
    assert not ok and any("not 'isaac-5.1'" in reason for reason in reasons)


def test_an_unrecorded_driver_version_is_a_rejection_not_a_skipped_check() -> None:
    # Publication is only granted on evidence. An attestation that carries no
    # driver version used to slip past the comparison entirely; now it is the
    # one driver outcome the gate names.
    backend = get_backend_descriptor("isaac-5.1")
    for missing in ("", "   "):
        attestation = _docker_attestation(driver_version=missing, publishable=False)
        assert attestation.driver_version == ""
        ok, reasons = evaluate_official_profile(attestation, backend)
        assert not ok
        assert reasons == ("driver version was not recorded",)


def test_a_recorded_string_that_is_not_a_version_is_also_a_rejection() -> None:
    backend = get_backend_descriptor("isaac-5.1")
    # The contract already refuses `publishable: true` beside such a string;
    # the gate must still name the reason on the unpublishable record.
    for junk in ("not-a-version", "535", "Driver Not Loaded", "v.x.y"):
        attestation = _docker_attestation(driver_version=junk, publishable=False)
        ok, reasons = evaluate_official_profile(attestation, backend)
        assert not ok, junk
        assert any("not a version" in reason for reason in reasons), reasons
        assert not any("not recorded" in reason for reason in reasons), reasons


def test_the_driver_version_is_recorded_not_compared_against_a_floor() -> None:
    # ADR 0008: min_driver is the vendor-tested minimum, kept as documentation.
    # The L20 ran Isaac Sim 5.1 on 535.216.03, below the listed 580.65.06.
    backend = get_backend_descriptor("isaac-5.1")
    assert backend.min_driver == "580.65.06"
    for driver in ("535.216.03", "580.65.06", "580.65.05", "1.0"):
        ok, reasons = evaluate_official_profile(_docker_attestation(driver_version=driver), backend)
        assert ok, (driver, reasons)
        assert reasons == ()


def test_the_digest_fields_gate_nothing() -> None:
    # Informational only: absent, locked, or disagreeing with a descriptor that
    # (hypothetically) carried one -- none of it is a reason.
    with_digest = dataclasses.replace(get_backend_descriptor("isaac-5.1"), image_digest=_DIGEST)
    for backend in (get_backend_descriptor("isaac-5.1"), with_digest):
        for attestation in (
            _docker_attestation(),
            _docker_attestation(image_digest=_DIGEST, digest_locked=True),
            _docker_attestation(image_digest="b" * 64, digest_locked=True),
            _docker_attestation(image_digest=_DIGEST, digest_locked=False),
        ):
            ok, reasons = evaluate_official_profile(attestation, backend)
            assert ok, reasons


def test_experimental_backend_results_are_never_publishable() -> None:
    backend = get_backend_descriptor("isaac-6.0")
    attestation = _docker_attestation(
        backend_id="isaac-6.0",
        backend_status="experimental",
        driver_version="595.58.03",
        publishable=False,
    )
    ok, reasons = evaluate_official_profile(attestation, backend)
    assert not ok and any("experimental" in reason for reason in reasons)
    # And the contract refuses the claim outright.
    with pytest.raises(ValidationError):
        _docker_attestation(backend_id="isaac-6.0", backend_status="experimental")


def test_a_run_from_todays_registry_state_can_publish() -> None:
    # Replaces the pre-ADR-0008 pin that nothing could pass this gate. With no
    # image digest published on any backend -- and none ever coming, since the
    # image embeds Isaac Sim -- a Linux run of the official backend with a
    # recorded driver is exactly what the leaderboard accepts.
    assert all(item.image_digest is None for item in SIM_BACKENDS)
    ok, reasons = evaluate_official_profile(
        _native_attestation(), get_backend_descriptor("isaac-5.1")
    )
    assert ok and reasons == ()


# --- the driver version is collected, not declared ---------------------------
#
# It was hard-coded to "" in the attestation producer, so every run looked exactly
# like a run on a machine with no driver. The L20 Track B runs are the concrete
# case: a real 535.216.03 that nothing recorded, so no record could say which
# driver its numbers came from -- and under ADR 0008 a recorded driver is exactly
# what separates a publishable Linux run from an unpublishable one.
#
# Collection is in-process NVML through ctypes -- no subprocess, no new dependency.
# The fakes below stand in for libnvidia-ml.so.1 so every branch is exercised on a
# machine that has no NVIDIA driver at all, which is where CI runs.


class _FakeNvml:
    """Stands in for libnvidia-ml.so.1 loaded through ctypes.CDLL.

    ``getattr`` is how the real ctypes library object exposes symbols, so a
    symbol this fake does not define raises AttributeError exactly as a .so
    missing that entry point would.
    """

    def __init__(self, *, version=b"535.216.03", init_rc=0, query_rc=0, symbols=None):
        self._version = version
        self._init_rc = init_rc
        self._query_rc = query_rc
        self._symbols = symbols
        self.init_calls = 0
        self.shutdown_calls = 0

    def __getattr__(self, name):
        if self._symbols is not None and name not in self._symbols:
            raise AttributeError(name)
        if name in ("nvmlInit_v2", "nvmlInit"):
            return self._init
        if name == "nvmlSystemGetDriverVersion":
            return self._query
        if name == "nvmlShutdown":
            return self._shutdown
        raise AttributeError(name)

    def _init(self):
        self.init_calls += 1
        return self._init_rc

    def _query(self, buffer, length):
        if self._query_rc == 0:
            buffer.value = self._version
        return self._query_rc

    def _shutdown(self):
        self.shutdown_calls += 1
        return 0


def _loader(library):
    """A ctypes.CDLL stand-in: returns *library*, or raises OSError if None."""

    def _load(name):
        if library is None:
            raise OSError(f"{name}: cannot open shared object file")
        return library

    return _load


def _no_files(monkeypatch, backend_mod):
    monkeypatch.setattr(backend_mod, "NVIDIA_DRIVER_VERSION_FILES", ("/nonexistent/version",))


def test_the_driver_version_is_read_from_nvml_in_process(monkeypatch) -> None:
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    nvml = _FakeNvml(version=b"535.216.03")
    monkeypatch.setattr(ctypes, "CDLL", _loader(nvml))

    assert backend_mod.observed_driver_version() == "535.216.03"
    assert nvml.init_calls == 1
    assert nvml.shutdown_calls == 1, "NVML must not be left initialised in a long-lived process"


def test_nvml_is_shut_down_even_when_the_query_fails(monkeypatch) -> None:
    # A failed query that leaked an initialised NVML would accumulate inside the
    # Isaac process, which outlives this call by an entire run.
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    nvml = _FakeNvml(query_rc=999)
    monkeypatch.setattr(ctypes, "CDLL", _loader(nvml))
    _no_files(monkeypatch, backend_mod)

    assert backend_mod.observed_driver_version() == ""
    assert nvml.shutdown_calls == 1


def test_a_machine_with_no_nvml_library_reports_unread(monkeypatch) -> None:
    """The CI case: no NVIDIA driver, so the .so does not exist."""
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    monkeypatch.setattr(ctypes, "CDLL", _loader(None))
    _no_files(monkeypatch, backend_mod)

    assert backend_mod.observed_driver_version() == ""


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"init_rc": 999}, "nvmlInit refused"),
        ({"query_rc": 999}, "nvmlSystemGetDriverVersion refused"),
        ({"symbols": {"nvmlShutdown"}}, "the .so exports no init symbol"),
        ({"symbols": {"nvmlInit_v2", "nvmlShutdown"}}, "the .so exports no query symbol"),
        ({"version": b"Driver Not Loaded"}, "the query answered with prose, not a version"),
        ({"version": b""}, "the query answered with nothing"),
        ({"version": b"\xff\xfe\x00garbage"}, "the buffer was not decodable text"),
        ({"version": b"535"}, "a bare integer is not a dotted driver version"),
        ({"version": b"v.x.y"}, "dotted, but not numeric"),
    ],
)
def test_an_nvml_api_failure_is_never_parsed_into_a_driver_version(
    monkeypatch, kwargs, why
) -> None:
    # A value that is not a version must not be recorded as one: the gate would
    # then report it as unparseable -- a claim something was read -- rather than
    # unreported, which is what happened.
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    monkeypatch.setattr(ctypes, "CDLL", _loader(_FakeNvml(**kwargs)))
    _no_files(monkeypatch, backend_mod)

    assert backend_mod.observed_driver_version() == "", why


class _RaisingNvml(_FakeNvml):
    """A library whose query raises instead of returning a status code."""

    def _query(self, buffer, length):
        raise ValueError("ctypes argument conversion failed")


def test_an_exception_inside_the_nvml_call_is_not_a_driver_version(monkeypatch) -> None:
    """A raising NVML must not take the run with it, and must not invent a value.

    ctypes surfaces a signature or conversion problem as an exception rather than
    a status code, so the status-code paths above do not cover it.
    """
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    nvml = _RaisingNvml()
    monkeypatch.setattr(ctypes, "CDLL", _loader(nvml))
    _no_files(monkeypatch, backend_mod)

    assert backend_mod.observed_driver_version() == ""
    # And the library is still released: the `finally` has to survive an
    # exception, not just a status code, or a raising query leaks an initialised
    # NVML into a process that outlives this call by a whole run.
    assert nvml.shutdown_calls == 1


def test_a_machine_with_no_nvml_falls_back_to_the_kernel_module(monkeypatch, tmp_path) -> None:
    # A container with the driver mounted but no NVML library. The prose form:
    # the number is embedded in a sentence, not alone on the line.
    import ctypes

    from insight_bench.simulator.isaac import backend as backend_mod

    version_file = tmp_path / "version"
    version_file.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.216.03  Fri Sep 20 00:00:00 UTC 2024\n"
    )
    monkeypatch.setattr(ctypes, "CDLL", _loader(None))
    monkeypatch.setattr(backend_mod, "NVIDIA_DRIVER_VERSION_FILES", (str(version_file),))

    assert backend_mod.observed_driver_version() == "535.216.03"


def test_collection_never_imports_ctypes_at_module_scope() -> None:
    """`import insight_bench` must not reach for anything driver-shaped.

    A module-level CDLL would run on every machine that imports the SDK,
    including the ones with no NVIDIA driver at all.
    """
    import inspect

    from insight_bench.simulator.isaac import backend as backend_mod

    source = inspect.getsource(backend_mod)
    header = source[: source.index("def _first_driver_version")]
    assert "import ctypes" not in header, "ctypes must be imported inside the function body"
    assert "import ctypes" in inspect.getsource(backend_mod._driver_version_from_nvml)


def test_the_real_producer_records_what_it_observed(monkeypatch) -> None:
    """The producer, not just the helper: what actually reaches the attestation.

    A helper that works while the attestation still writes "" would leave the
    original defect exactly where it was.
    """
    from insight_bench.simulator.isaac import backend as backend_mod

    # The attestation refuses to describe an unsupported platform, and CI is macOS.
    # Only the OS probe is stood in for; the driver path under test is untouched.
    monkeypatch.setattr(backend_mod, "_observed_os", lambda: "linux")
    monkeypatch.setattr(backend_mod, "_in_docker", lambda: False)
    monkeypatch.setattr(backend_mod, "observed_driver_version", lambda: "535.216.03")
    produced = backend_mod.IsaacCameraWalkBackend().runtime_attestation()

    assert produced.driver_version == "535.216.03"
    assert (produced.runtime_kind, produced.os) == ("native", "linux")
    assert produced.publishable is True, "an attested native Linux run publishes (ADR 0008)"
    assert produced.image_digest is None and produced.digest_locked is False


def test_the_real_producer_reports_an_unreadable_driver_as_unread(monkeypatch) -> None:
    from insight_bench.simulator.isaac import backend as backend_mod

    monkeypatch.setattr(backend_mod, "_observed_os", lambda: "linux")
    monkeypatch.setattr(backend_mod, "observed_driver_version", lambda: "")
    produced = backend_mod.IsaacCameraWalkBackend().runtime_attestation()

    assert produced.driver_version == ""
    assert produced.publishable is False, "an unrecorded driver keeps a Linux run out"


def test_the_producers_verdict_is_the_gates_verdict(monkeypatch) -> None:
    """End to end: producer -> attestation -> official gate, on the real L20 driver."""
    from insight_bench.simulator.isaac import backend as backend_mod

    monkeypatch.setattr(backend_mod, "_observed_os", lambda: "linux")
    monkeypatch.setattr(backend_mod, "observed_driver_version", lambda: "535.216.03")
    produced = backend_mod.IsaacCameraWalkBackend().runtime_attestation()
    backend = get_backend_descriptor("isaac-5.1")

    ok, reasons = evaluate_official_profile(produced, backend)
    assert ok, reasons
    assert produced.publishable is ok
    # Nothing mentions the vendor minimum: it is documentation, not a floor.
    assert backend.min_driver not in " | ".join(reasons)


def test_an_unread_driver_keeps_the_gate_fail_closed() -> None:
    # Publication must stay withheld when the question could not be answered.
    # An unreadable driver is a rejection, never a skipped check.
    backend = get_backend_descriptor("isaac-5.1")
    ok, reasons = evaluate_official_profile(
        _docker_attestation(driver_version="", publishable=False), backend
    )

    assert not ok
    assert reasons == ("driver version was not recorded",)


def test_the_l20_driver_below_the_vendor_minimum_is_publishable() -> None:
    """The acceptance case ADR 0008 was written for: 535.216.03 on the L20.

    Isaac Sim 5.1 demonstrably ran there, below NVIDIA's listed 580.65.06, so
    the version is recorded as evidence and compared against nothing.
    """
    backend = get_backend_descriptor("isaac-5.1")
    attestation = _docker_attestation(driver_version="535.216.03")

    ok, reasons = evaluate_official_profile(attestation, backend)
    assert ok, reasons
    assert attestation.driver_version == "535.216.03"
    assert backend.min_driver == "580.65.06"


def test_the_attestation_records_the_isaac_stack_it_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run on Isaac Lab 0.46.3 must not leave the same record as one on the
    declared 2.3.0. The versions are read off the machine and recorded, on the
    same terms as the driver: evidence, compared against nothing.
    """
    monkeypatch.setattr(
        isaac_backend,
        "observed_isaac_versions",
        lambda: {
            "isaac_lab_version": "0.46.3",
            "isaac_sim_version": "5.1.0-rc.6",
            "python_version": "3.11.14",
        },
    )
    monkeypatch.setattr(isaac_backend, "observed_driver_version", lambda: "535.216.03")
    monkeypatch.setattr(isaac_backend, "_observed_os", lambda: "linux")

    record = isaac_backend.IsaacCameraWalkBackend().runtime_attestation()

    assert record.isaac_lab_version == "0.46.3"
    assert record.isaac_sim_version == "5.1.0-rc.6"
    assert record.python_version == "3.11.14"
    # Recorded, not compared: the declared line says 2.3.0 and the run is still
    # publishable on the ADR 0008 terms -- official backend, linux, a driver.
    assert default_backend().isaac_lab == "2.3.0"
    assert record.publishable is True


def test_an_unreadable_isaac_version_is_recorded_as_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``""`` is the vocabulary for "not recorded" -- no ``unknown``, no guess."""
    monkeypatch.setattr(
        isaac_backend,
        "observed_isaac_versions",
        lambda: {"isaac_lab_version": "", "isaac_sim_version": "", "python_version": "3.11.14"},
    )
    monkeypatch.setattr(isaac_backend, "observed_driver_version", lambda: "535.216.03")
    monkeypatch.setattr(isaac_backend, "_observed_os", lambda: "linux")

    record = isaac_backend.IsaacCameraWalkBackend().runtime_attestation()

    assert record.isaac_lab_version == ""
    assert record.isaac_sim_version == ""
    # An unread Isaac version is not the one outcome the gate rejects; that is
    # still the driver alone (ADR 0008).
    assert record.publishable is True


def test_reading_the_isaac_stack_never_raises_into_a_run() -> None:
    """The real reader, on a machine with no Isaac at all: three keys, no throw."""
    observed = isaac_backend.observed_isaac_versions()

    assert set(observed) == {"isaac_lab_version", "isaac_sim_version", "python_version"}
    assert observed["python_version"]  # always readable
