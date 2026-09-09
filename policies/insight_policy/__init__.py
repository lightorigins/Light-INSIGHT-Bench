"""Model-side SDK for serving a navigation policy to INSIGHT-Bench.

This package runs inside the MODEL's own Python environment, next to whatever
torch / transformers pins that model needs. It therefore depends on nothing but
the standard library, ``numpy`` and ``pillow``, parses under Python 3.8 (several
baselines pin py3.8/3.9 venvs), and never imports ``insight_bench``: the
evaluation SDK and the model live in separate environments and speak only the
four-endpoint HTTP protocol implemented in :mod:`insight_policy.server`.

Write a policy by subclassing :class:`Policy` and implementing three methods::

    class MyPolicy(Policy):
        model_id = "my-org/my-model"

        def __init__(self, model_path, **options):
            super().__init__(model_path, **options)
            self.model = load_my_model(model_path)

        def reset(self, instruction): ...
        def act(self, rgb): return Step.primitives(["forward"])
        def finish(self): ...

Then serve it::

    python -m insight_policy --policy policies/my_policy --model-path /ckpt

``act`` receives one HWC uint8 RGB frame (480x270 as the benchmark rig captures
it, smaller if this policy asks for a centre crop or a resize) and returns a
:class:`Step`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "PRIMITIVE_NAMES",
    "Policy",
    "Step",
    "Waypoint",
    "coerce_option",
    "resolve_defaults",
]

# The three discrete moves a policy may name instead of computing displacements.
# The server expands them at this policy's ``forward_m`` / ``turn_deg``.
PRIMITIVE_NAMES = ("forward", "left", "right")


@dataclass(frozen=True)
class Waypoint:
    """One body-frame displacement.

    ``forward_m`` is along the camera's optical axis, ``lateral_m`` is positive
    to the left, and ``yaw_rad`` is positive counter-clockwise (= turn left):
    the benchmark integrates poses in a right-handed z-up frame, so a positive
    yaw and a positive forward move the camera the way a human would expect.

    Two limits of the published suite are worth knowing before you use the third
    degree of freedom. It drops lateral velocity, and it clips forward velocity
    to a non-negative range. So a motion that is lateral-only, or that walks
    backwards, executes as standing still -- and the first command that executes
    as standing still ENDS the episode. Turn, then walk.
    """

    forward_m: float = 0.0
    lateral_m: float = 0.0
    yaw_rad: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "forward_m": float(self.forward_m),
            "lateral_m": float(self.lateral_m),
            "yaw_rad": float(self.yaw_rad),
        }


def _as_waypoints(deltas: Any) -> tuple[Waypoint, ...]:
    """Normalize whatever ``Step.waypoints`` was handed into ``Waypoint`` objects.

    Accepts a single ``Waypoint``, a sequence of them, a numpy array shaped
    (N, 3) or (3,), or any sequence of ``(forward_m, lateral_m, yaw_rad)``.
    """
    if isinstance(deltas, Waypoint):
        return (deltas,)
    if hasattr(deltas, "tolist"):  # numpy array / torch tensor already on cpu
        rows: Any = deltas.tolist()
    else:
        rows = list(deltas)
    # (a, b) rather than a | b in isinstance: this package parses under py3.8.
    if rows and not isinstance(rows[0], (list, tuple, Waypoint)):  # noqa: UP038
        rows = [rows]  # a bare (forward_m, lateral_m, yaw_rad) triple
    out = []
    for row in rows:
        if isinstance(row, Waypoint):
            out.append(row)
            continue
        values = list(row)
        if len(values) != 3:
            raise ValueError(
                "each waypoint must be (forward_m, lateral_m, yaw_rad); "
                f"got {len(values)} values: {values!r}"
            )
        out.append(Waypoint(float(values[0]), float(values[1]), float(values[2])))
    return tuple(out)


@dataclass(frozen=True)
class Step:
    """What one call to :meth:`Policy.act` decided.

    Build one with a constructor, never with the fields directly::

        Step.waypoints([(0.25, 0.0, 0.0)])   # explicit displacements
        Step.primitives(["left", "forward"]) # named moves, server expands them
        Step.stop()                          # "I have arrived"

    A step carrying several waypoints or primitives is fine: the server queues
    them and releases exactly one per ``/act``, so the simulator keeps a frame
    per executed motion.
    """

    waypoint_deltas: tuple[Waypoint, ...] = ()
    primitive_names: tuple[str, ...] = ()
    stop_requested: bool = False
    raw_output: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def waypoints(
        cls,
        deltas: Any,
        *,
        then_stop: bool = False,
        raw_output: str = "",
        extras: dict[str, Any] | None = None,
    ) -> Step:
        """Displacements in metres/radians, as an array or list of triples.

        ``then_stop=True`` for a chunk that ends the episode after its last
        motion -- the queue drains first, and the reply after it is the stop.
        """
        return cls(
            waypoint_deltas=_as_waypoints(deltas),
            stop_requested=bool(then_stop),
            raw_output=raw_output,
            extras=dict(extras or {}),
        )

    @classmethod
    def primitives(
        cls,
        names: Iterable[str],
        *,
        then_stop: bool = False,
        raw_output: str = "",
        extras: dict[str, Any] | None = None,
    ) -> Step:
        """Named moves from :data:`PRIMITIVE_NAMES`, expanded by the server.

        ``then_stop=True`` for an action chunk whose last token was "stop": the
        moves before it still execute, one per /act, and the stop follows them.
        """
        chosen = tuple(str(name) for name in names)
        for name in chosen:
            if name not in PRIMITIVE_NAMES:
                raise ValueError(f"unknown primitive {name!r}; expected one of {PRIMITIVE_NAMES}")
        return cls(
            primitive_names=chosen,
            stop_requested=bool(then_stop),
            raw_output=raw_output,
            extras=dict(extras or {}),
        )

    @classmethod
    def stop(
        cls,
        *,
        raw_output: str = "",
        extras: dict[str, Any] | None = None,
    ) -> Step:
        """End the episode here. The benchmark scores the pose of this frame."""
        return cls(stop_requested=True, raw_output=raw_output, extras=dict(extras or {}))

    def is_empty(self) -> bool:
        """No motion and no stop: the model said nothing usable this frame."""
        return not self.waypoint_deltas and not self.primitive_names and not self.stop_requested


def resolve_defaults(policy_cls: type) -> dict[str, Any]:
    """Every option this policy class declares, base classes first.

    A subclass' ``defaults`` block extends its parent's rather than replacing
    it, so a model only writes down the constants that differ from the base.
    """
    merged: dict[str, Any] = {}
    for klass in reversed(policy_cls.__mro__):
        merged.update(vars(klass).get("defaults") or {})
    return merged


def _coerce_bool(name: str, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"option {name}={raw!r} is not a boolean")


def _coerce_sequence(raw: str) -> tuple[float, ...] | tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",")]
    if all(part.lstrip("-").isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return tuple(float(part) for part in parts)


def coerce_option(name: str, default: Any, raw: Any) -> Any:
    """Turn one ``--opt key=value`` string into the type the default implies.

    Non-string values pass straight through, so constructing a policy in Python
    with real ints and floats needs no ceremony. ``none`` (any case) and the
    empty string always mean ``None``.
    """
    if not isinstance(raw, str):
        return raw
    if raw.strip().lower() in ("none", "null", ""):
        return None
    if isinstance(default, bool):
        return _coerce_bool(name, raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, (list, tuple)):  # noqa: UP038
        return _coerce_sequence(raw)
    if isinstance(default, str):
        return raw
    # default is None: the declared type is unknown, so try the ladder the
    # nullable options in this SDK actually use (hfov floats, resize pairs).
    if "," in raw:
        return _coerce_sequence(raw)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


class Policy:
    """Base class for a navigation policy. Subclass it; implement three methods.

    ``model_id`` identifies the checkpoint on the wire and in ``/health``; it is
    what a leaderboard entry is attributed to, so set it to the real published
    identifier of the weights you are serving.

    ``defaults`` declares every option this policy accepts, with its default
    value. The CLI's ``--opt key=value`` may override any of them and nothing
    else -- an unknown key is an error rather than a silently ignored typo. The
    block below is the part the SERVER reads; a model adds its own knobs to it.
    """

    model_id: ClassVar[str] = "TODO/unknown-model"

    defaults: ClassVar[dict[str, Any]] = {
        # Size of one "forward" / "left" / "right" primitive. Also the fallback
        # step size when on_empty == "forward".
        "forward_m": 0.25,
        "turn_deg": 30.0,
        # Feed frames observed while queued motions are draining back to the
        # policy via observe()? Models whose upstream eval loop counts every
        # simulator frame in their history want this; models that run open-loop
        # inside an action chunk do not.
        "queue_feed_frames": False,
        # What to do when act() returns nothing usable: "forward" keeps the
        # episode alive (recoverable parse failure), "stop" ends it.
        "on_empty": "forward",
        # Frame preprocessing. The benchmark rig is 480x270 at 120 deg HFOV; a
        # model trained at a narrower HFOV can ask for a centre crop to its own
        # training field of view, optionally at a fixed width/height aspect,
        # and optionally a resize to (height, width).
        "source_hfov_deg": 120.0,
        "center_crop_hfov_deg": None,
        "crop_aspect": None,
        "resize": None,
    }

    def __init__(self, model_path: str, **options: Any):
        self.model_path = str(model_path)
        declared = resolve_defaults(type(self))
        unknown = sorted(set(options) - set(declared))
        if unknown:
            known = ", ".join(sorted(declared))
            raise ValueError(
                f"{type(self).__name__} does not accept option(s) {unknown}; known: {known}"
            )
        resolved = dict(declared)
        resolved.update(options)
        self.options: dict[str, Any] = resolved

    # ---- the three methods a policy implements ---------------------------------------

    def reset(self, instruction: str) -> None:
        """Start a new episode. Drop all per-episode state; keep the weights."""
        raise NotImplementedError

    def act(self, rgb: Any) -> Step:
        """Decide the next move from one HWC uint8 RGB frame."""
        raise NotImplementedError

    def finish(self) -> None:
        """The episode is over. Release per-episode state. Optional."""

    # ---- optional hooks --------------------------------------------------------------

    def observe(self, rgb: Any) -> None:
        """A frame the simulator rendered while a queued motion was executing.

        Only called when this policy sets ``queue_feed_frames: True``, and never
        expected to run inference -- append the frame to history and return.
        """

    def describe(self) -> dict[str, Any]:
        """Extra ``/health`` metadata: configuration, not counters.

        Read once, before the first episode, so a counter here is published as
        zero however many steps follow.
        """
        return {}

    def healthy(self) -> bool:
        """Whether this policy can still answer. ``/health`` reports it.

        Override it when the model can go away while the server stays up -- a
        policy running the model in a child process, most obviously. One that
        did not, and answered every ``/act`` with something that decoded to no
        motion, spent two full 300-action budgets producing zero displacement
        while ``/health`` said ``healthy: true``. Nothing refused it, because
        the only liveness anybody could see was the server's own.
        """
        return True

    def close(self) -> None:
        """Release anything the policy holds. The server calls it on shutdown.

        Override it when loading the model took more than memory -- a child
        process, most obviously. One that did not leaked its child on every
        run: the orphan kept the fixed port it had bound, and the *next* run's
        child then died at startup on "address already in use", which is a
        failure that shows up one run after the run that caused it.
        """

    # ---- option readers --------------------------------------------------------------

    def opt_float(self, name: str) -> float:
        return float(self.options[name])

    def opt_int(self, name: str) -> int:
        return int(self.options[name])

    def opt_bool(self, name: str) -> bool:
        return bool(self.options[name])

    def opt_str(self, name: str) -> str:
        value = self.options[name]
        return "" if value is None else str(value)

    def opt_path(self, name: str) -> str:
        """A required filesystem option, checked early so failures are readable."""
        value = self.opt_str(name)
        if not value:
            raise ValueError(
                f"{type(self).__name__} needs --opt {name}=<path>; it has no public default"
            )
        if not Path(value).expanduser().exists():
            raise ValueError(f"--opt {name}={value!r} does not exist")
        return str(Path(value).expanduser())
