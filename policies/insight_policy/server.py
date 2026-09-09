"""The four-endpoint HTTP protocol INSIGHT-Bench speaks to a policy.

Built on ``http.server`` on purpose: this code has to run inside a baseline's
own venv, and several of those pin Python 3.8 with transformers 4.31-era
dependency sets that cannot install pydantic 2 / fastapi. The stdlib always can.

The wire contract (fixed; the simulator side is the authority)::

    GET  /health  -> {"healthy": true, "model_id": "...", ...}
    POST /reset   {"instruction", "episode_id"} -> {"success": true}
    POST /act     {"episode_id", "frame_id", "timestamp_ms", "image_jpeg_b64"}
                  -> {"success": true, "action_name": "WAYPOINT"|"stop",
                      "waypoint": {"cluster_id", "waypoints": [...]}, ...}
    POST /finish  {"episode_id"} -> {"success": true}

Failures are HTTP 200 with ``{"success": false, "error": "..."}``. A 5xx would
make the runner's HTTP client raise before it could read the reason, so every
handler funnels exceptions into that shape instead.

This module owns everything between the socket and :meth:`Policy.act`:

* per-episode sessions keyed by ``episode_id``;
* the two shapes an answer comes in: a waypoint chunk is a PLAN and crosses the
  wire whole, for the runner to pick a row from and the model to re-plan next
  frame; named primitives are separate moves and queue, one released per ``/act``
  so the simulator renders a frame per motion;
* splitting a queued primitive to the per-step execution caps the runner enforces;
* the optional centre-crop + resize into a model's training field of view;
* the ``on_empty`` fallback when a policy returns nothing usable;
* JPEG decode and readable errors for an episode that was never reset.
"""

from __future__ import annotations

import base64
import io
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from insight_policy import PRIMITIVE_NAMES, Policy, Step, Waypoint

# Per-control-step execution caps. The runner turns one waypoint into a velocity
# command held for control_dt_sec and clips it to the suite's velocity ranges --
# insight_bench is control_dt_sec 0.1 with lin_vel_range_mps (0.0, 2.5) and
# ang_vel_range_dps (-300.0, 300.0), i.e. at most 0.25 m or 30 deg of motion
# actually executes per /act. Anything larger would be silently clipped, so a
# larger motion is split across consecutive replies instead.
MAX_TRANSLATION_M_PER_STEP = 0.25
MAX_TURN_DEG_PER_STEP = 30.0
MAX_TURN_RAD_PER_STEP = math.radians(MAX_TURN_DEG_PER_STEP)

# Below these magnitudes a waypoint is a no-op, and a no-op is dangerous: the
# first command under the suite's zero-velocity threshold ends the episode. Such
# waypoints are dropped rather than sent.
NOOP_TRANSLATION_ATOL_M = 0.005
NOOP_TURN_ATOL_RAD = math.radians(0.05)

# Most policies emit geometry directly, not an index into a trajectory
# vocabulary, so there is no cluster to report. -1 is the sentinel the trace
# schema already carries for "this response came from the discrete-action path".
# A policy that DOES decode from a vocabulary reports its codeword by putting an
# integer ``cluster_id`` in its Step extras; see :func:`_cluster_id_from_extras`.
NO_CLUSTER_ID = -1

_DEFAULT_SESSION_KEY = "__default__"

WAYPOINT_ACTION_NAME = "WAYPOINT"
STOP_ACTION_NAME = "stop"


# ---------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------


def _split_scalar(value: float, cap: float) -> list[float]:
    """Split ``value`` into chunks of magnitude <= cap, preserving sign and total."""
    if abs(value) <= cap:
        return [value]
    sign = 1.0 if value >= 0 else -1.0
    magnitude = abs(value)
    chunks = []
    while magnitude > cap:
        chunks.append(sign * cap)
        magnitude -= cap
    if magnitude > 0.0:
        chunks.append(sign * magnitude)
    return chunks


def _split_translation(forward: float, lateral: float, cap: float) -> list[tuple[float, float]]:
    """Split a 2-D translation into capped chunks, preserving its direction."""
    magnitude = math.hypot(forward, lateral)
    if magnitude <= cap:
        return [(forward, lateral)]
    unit_f, unit_l = forward / magnitude, lateral / magnitude
    chunks = []
    remaining = magnitude
    while remaining > cap:
        chunks.append((unit_f * cap, unit_l * cap))
        remaining -= cap
    if remaining > 0.0:
        chunks.append((unit_f * remaining, unit_l * remaining))
    return chunks


def fits_one_step(waypoint: Waypoint) -> bool:
    """Can the runner execute this whole waypoint in a single control step?"""
    return (
        math.hypot(waypoint.forward_m, waypoint.lateral_m) <= MAX_TRANSLATION_M_PER_STEP
        and abs(waypoint.yaw_rad) <= MAX_TURN_RAD_PER_STEP
    )


def split_waypoint(waypoint: Waypoint) -> list[Waypoint]:
    """One waypoint -> the cap-compliant single-step waypoints that realise it.

    A waypoint that already fits inside one step is passed through untouched,
    curve and all: a policy that asked to turn and translate together gets
    exactly the motion it asked for.

    One that does not fit has to become several replies, and there the turn goes
    first and the walk after. Interleaving them would trace an arc the policy
    never asked for, and it is what the discrete baselines mean anyway -- their
    text parsers produce things like "turn left 45 and go forward 1 m".
    """
    if fits_one_step(waypoint):
        return [waypoint]
    steps: list[Waypoint] = []
    for yaw in _split_scalar(waypoint.yaw_rad, MAX_TURN_RAD_PER_STEP):
        if abs(yaw) >= NOOP_TURN_ATOL_RAD:
            steps.append(Waypoint(yaw_rad=yaw))
    for forward, lateral in _split_translation(
        waypoint.forward_m, waypoint.lateral_m, MAX_TRANSLATION_M_PER_STEP
    ):
        if math.hypot(forward, lateral) >= NOOP_TRANSLATION_ATOL_M:
            steps.append(Waypoint(forward_m=forward, lateral_m=lateral))
    return steps


def is_noop(waypoint: Waypoint) -> bool:
    return (
        math.hypot(waypoint.forward_m, waypoint.lateral_m) < NOOP_TRANSLATION_ATOL_M
        and abs(waypoint.yaw_rad) < NOOP_TURN_ATOL_RAD
    )


def normalize_waypoints(waypoints: list[Waypoint]) -> list[Waypoint]:
    """Split to the per-step caps and drop no-ops."""
    steps: list[Waypoint] = []
    for waypoint in waypoints:
        if is_noop(waypoint):
            continue
        steps.extend(split_waypoint(waypoint))
    return steps


def primitive_to_waypoint(name: str, *, forward_m: float, turn_deg: float) -> Waypoint:
    """Expand a named primitive at this policy's step size."""
    if name == "forward":
        return Waypoint(forward_m=float(forward_m))
    if name == "left":
        return Waypoint(yaw_rad=math.radians(float(turn_deg)))
    if name == "right":
        return Waypoint(yaw_rad=-math.radians(float(turn_deg)))
    raise ValueError(f"unknown primitive {name!r}; expected one of {PRIMITIVE_NAMES}")


# ---------------------------------------------------------------------------------
# frame preprocessing
# ---------------------------------------------------------------------------------


class FramePreprocessor:
    """Optional centre-crop from the rig's HFOV to a model's training HFOV.

    ``crop_w = W * tan(target/2) / tan(source/2)``; the height is then cropped to
    ``crop_aspect`` (width / height) when one is set, so a model trained on 4:3
    frames gets a 4:3 crop whose vertical field of view also matches. ``resize``
    is (height, width), applied after the crop.

    A crop changes what the model can see, so it is a declared property of the
    policy that ends up in ``/health`` and therefore in the run's evidence -- not
    something the harness does behind the model's back.
    """

    def __init__(self, options: dict[str, Any]):
        self.source_hfov_deg = float(options.get("source_hfov_deg") or 120.0)
        crop = options.get("center_crop_hfov_deg")
        self.center_crop_hfov_deg = None if crop is None else float(crop)
        aspect = options.get("crop_aspect")
        self.crop_aspect = None if aspect is None else float(aspect)
        resize = options.get("resize")
        self.resize = None if resize is None else (int(resize[0]), int(resize[1]))

    @property
    def enabled(self) -> bool:
        return self.center_crop_hfov_deg is not None or self.resize is not None

    def describe(self) -> dict[str, Any]:
        return {
            "source_hfov_deg": self.source_hfov_deg,
            "center_crop_hfov_deg": self.center_crop_hfov_deg,
            "crop_aspect": self.crop_aspect,
            "resize": list(self.resize) if self.resize else None,
        }

    def __call__(self, frame_rgb: Any) -> Any:
        frame = frame_rgb
        if self.center_crop_hfov_deg is not None:
            height, width = frame.shape[:2]
            ratio = math.tan(math.radians(self.center_crop_hfov_deg) / 2.0) / math.tan(
                math.radians(self.source_hfov_deg) / 2.0
            )
            crop_w = max(2, min(width, round(width * ratio)))
            crop_h = height
            if self.crop_aspect is not None:
                crop_h = max(2, min(height, round(crop_w / self.crop_aspect)))
            x0 = (width - crop_w) // 2
            y0 = (height - crop_h) // 2
            frame = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        if self.resize is not None:
            from PIL import Image

            target_h, target_w = self.resize
            frame = np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR))
        return frame


def decode_jpeg_b64(data: str) -> Any:
    """Base64 JPEG on the wire -> HWC uint8 RGB array."""
    from PIL import Image

    if not data:
        raise ValueError("image_jpeg_b64 must not be empty")
    raw = base64.b64decode(data, validate=True)
    image = Image.open(io.BytesIO(raw))
    return np.asarray(image.convert("RGB"))


# ---------------------------------------------------------------------------------
# session bookkeeping and protocol handlers
# ---------------------------------------------------------------------------------


@dataclass
class _Session:
    episode_id: str
    instruction: str
    queue: deque = field(default_factory=deque)
    pending_stop: bool = False
    inference_calls: int = 0
    act_calls: int = 0
    # Codeword the last inference reported, so a queue drained over several
    # replies keeps naming the vocabulary entry its motions came from.
    cluster_id: int = NO_CLUSTER_ID


def _cluster_id_from_extras(extras: dict[str, Any]) -> int:
    """The trajectory-vocabulary codeword a policy reported, or the sentinel.

    The runner reads ``waypoint_cluster_id`` straight into the step record, so a
    policy that decodes from a vocabulary (LightNav-0's RVQ codes) has to be able
    to put its real codeword there instead of the discrete-action sentinel.
    """
    try:
        return int(extras["cluster_id"])
    except (KeyError, TypeError, ValueError):
        return NO_CLUSTER_ID


def _merge_extra(source: str, session: _Session, extras: dict[str, Any]) -> dict[str, Any]:
    """Bookkeeping the harness records, plus whatever the policy attached."""
    merged = {
        "source": source,
        "queue_len": len(session.queue),
        "inference_calls": session.inference_calls,
    }
    merged.update(extras)
    return merged


class PolicyHost:
    """Protocol logic, independent of the HTTP layer so it can be tested directly."""

    def __init__(self, policy: Policy):
        self.policy = policy
        options = policy.options
        self.forward_m = float(options["forward_m"])
        self.turn_deg = float(options["turn_deg"])
        self.queue_feed_frames = bool(options["queue_feed_frames"])
        self.on_empty = str(options["on_empty"])
        if self.on_empty not in ("forward", "stop"):
            raise ValueError(f"on_empty must be 'forward' or 'stop', got {self.on_empty!r}")
        self.preprocessor = FramePreprocessor(options)
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        # A Policy has no per-session state of its own -- reset() takes only an
        # instruction -- so exactly one episode can be in flight at a time. A
        # second /reset takes over; /act on the displaced episode is an error
        # rather than a silent read of the wrong episode's model state.
        self._active_key: str | None = None

    @staticmethod
    def _session_key(episode_id: str) -> str:
        return episode_id or _DEFAULT_SESSION_KEY

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = None if self._active_key is None else self._sessions.get(self._active_key)
            # The policy's own answer, not the server's. A server whose model
            # has died still accepts connections, and saying `true` here let a
            # run spend its whole action budget on a policy that could not act.
            try:
                policy_healthy = bool(self.policy.healthy())
            except Exception:
                policy_healthy = False
            payload: dict[str, Any] = {
                "healthy": policy_healthy,
                "model_id": self.policy.model_id,
                "model_path": self.policy.model_path,
                "policy_class": type(self.policy).__name__,
                "forward_m": self.forward_m,
                "turn_deg": self.turn_deg,
                "queue_feed_frames": self.queue_feed_frames,
                "on_empty": self.on_empty,
                "preprocess": self.preprocessor.describe(),
                "num_active_sessions": len(self._sessions),
                "active_episode_id": active.episode_id if active is not None else "",
            }
            for key, value in self.policy.describe().items():
                payload["policy_" + str(key)] = value
            return payload

    def reset(self, instruction: str, *, episode_id: str = "") -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        with self._lock:
            started = time.monotonic()
            key = self._session_key(episode_id)
            self._sessions[key] = _Session(episode_id=episode_id, instruction=instruction)
            self._active_key = key
            self.policy.reset(instruction)
            elapsed_ms = (time.monotonic() - started) * 1000.0
        return {
            "success": True,
            "episode_id": episode_id,
            "timings_ms": {"policy_reset_ms": elapsed_ms},
        }

    def finish(self, *, episode_id: str = "") -> dict[str, Any]:
        with self._lock:
            key = self._session_key(episode_id)
            removed = self._sessions.pop(key, None) is not None
            if key == self._active_key:
                # Only the live episode's model state may be torn down; a late
                # /finish for a displaced episode must not wipe the new one.
                self._active_key = None
                self.policy.finish()
            return {
                "success": True,
                "episode_id": episode_id,
                "removed": removed,
                "num_active_sessions": len(self._sessions),
            }

    def act(self, rgb_frame: Any, *, frame_id: int = 0, episode_id: str = "") -> dict[str, Any]:
        with self._lock:
            session = self._require_session(episode_id)
            session.act_calls += 1

            started = time.monotonic()
            source = "queue"
            raw_output = ""
            extras: dict[str, Any] = {}
            plan: list[Waypoint] = []

            if not session.queue and not session.pending_stop:
                source = "inference"
                frame = self.preprocessor(rgb_frame) if self.preprocessor.enabled else rgb_frame
                step = self.policy.act(frame)
                if not isinstance(step, Step):
                    raise TypeError(
                        f"{type(self.policy).__name__}.act must return a Step, got "
                        f"{type(step).__name__}"
                    )
                session.inference_calls += 1
                raw_output = step.raw_output
                extras = dict(step.extras)
                session.cluster_id = _cluster_id_from_extras(extras)
                if step.stop_requested:
                    session.pending_stop = True
                if step.waypoint_deltas:
                    # A waypoint chunk is a PLAN, handed over whole. The runner
                    # picks the row it executes and the model re-plans on the
                    # next frame, which is what a trajectory model's own
                    # evaluation loop does.
                    #
                    # Releasing the rows one per /act instead would be wrong
                    # twice over: the model would decide once every H frames
                    # rather than every frame, and a chunk's rows are usually
                    # cumulative poses, so executing row i as one step applies i
                    # steps of motion at once. Measured on LightNav-0, that made
                    # it turn on the spot for a whole episode and never advance.
                    plan = list(step.waypoint_deltas)
                elif step.primitive_names:
                    # The other kind of answer: a model that emits "forward,
                    # left, left" means three separate moves, and its own
                    # evaluation executes them one per frame. Those queue, and
                    # each is capped to one step's worth of motion.
                    session.queue.extend(
                        normalize_waypoints(
                            [
                                primitive_to_waypoint(
                                    name, forward_m=self.forward_m, turn_deg=self.turn_deg
                                )
                                for name in step.primitive_names
                            ]
                        )
                    )
                if not plan and not session.queue and not session.pending_stop:
                    # The fallback goes through the same normalisation as a real
                    # motion. A policy with forward_m above the per-step cap would
                    # otherwise emit one oversized reply that the runner silently
                    # clips -- the exact failure the caps exist to prevent.
                    fallback = (
                        []
                        if self.on_empty == "stop"
                        else normalize_waypoints([Waypoint(forward_m=self.forward_m)])
                    )
                    session.queue.extend(fallback)
                    if not fallback:
                        session.pending_stop = True
                    extras["on_empty_fallback"] = self.on_empty
            elif session.queue and self.queue_feed_frames:
                frame = self.preprocessor(rgb_frame) if self.preprocessor.enabled else rgb_frame
                self.policy.observe(frame)

            elapsed_ms = (time.monotonic() - started) * 1000.0
            # Only a step that actually ran the model reports an inference time.
            # A queued release used to report the microsecond it took to pop the
            # deque, which made "was the model asked on this step?" answerable
            # only by comparing magnitudes: the obvious check -- is the key
            # present, or above zero -- called a policy that never decoded
            # perfectly healthy, which is exactly how a one-decode-per-ten-steps
            # bug once went unnoticed.
            base: dict[str, Any] = {
                "success": True,
                "episode_id": session.episode_id,
                "frame_id": int(frame_id),
                "raw_output": raw_output,
                "inference_time_ms": elapsed_ms if source == "inference" else None,
                "timings_ms": {
                    "inference_total_ms": elapsed_ms if source == "inference" else None,
                    "act_total_ms": elapsed_ms,
                },
                "extra": _merge_extra(source, session, extras),
            }

            if plan:
                base["action_name"] = WAYPOINT_ACTION_NAME
                base["waypoint"] = {
                    "cluster_id": session.cluster_id,
                    "waypoints": [waypoint.to_dict() for waypoint in plan],
                }
                base["waypoint_cluster_id"] = session.cluster_id
                base["waypoint_horizon"] = len(plan)
                return base

            if session.queue:
                waypoint = session.queue.popleft()
                base["action_name"] = WAYPOINT_ACTION_NAME
                base["waypoint"] = {
                    "cluster_id": session.cluster_id,
                    "waypoints": [waypoint.to_dict()],
                }
                base["waypoint_cluster_id"] = session.cluster_id
                base["waypoint_horizon"] = 1
                return base

            # An empty queue here implies pending_stop: on_empty guarantees one
            # or the other, so a reply that moves nowhere is always a real stop.
            base["action_name"] = STOP_ACTION_NAME
            return base

    def _require_session(self, episode_id: str) -> _Session:
        key = self._session_key(episode_id)
        if not self._sessions:
            raise RuntimeError("must call /reset before /act")
        session = self._sessions.get(key)
        if session is None:
            known = sorted(s.episode_id for s in self._sessions.values())
            raise RuntimeError(f"episode_id has not been reset: {episode_id!r}; reset={known}")
        if key != self._active_key:
            active = None if self._active_key is None else self._sessions.get(self._active_key)
            active_id = active.episode_id if active is not None else None
            raise RuntimeError(
                "policy serves one episode at a time; " f"active={active_id!r} got={episode_id!r}"
            )
        return session


# ---------------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------------


def _text_field(payload: dict[str, Any], name: str, default: str = "") -> str:
    value = payload.get(name, default)
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _jpeg_field(payload: dict[str, Any], *names: str) -> str:
    """The first present image field. Several spellings exist in the wild."""
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, bytes) and value:
            return base64.b64encode(value).decode("ascii")
    return ""


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type, host: PolicyHost):
        super().__init__(address, handler)
        self.host_impl = host


class _Handler(BaseHTTPRequestHandler):
    # Keep-alive matters: an episode is up to 300 /act calls and the runner uses
    # a pooled HTTP session. HTTP/1.1 is safe here because every reply below
    # sends an accurate Content-Length.
    protocol_version = "HTTP/1.1"
    server_version = "insight-policy"
    sys_version = ""

    @property
    def _host(self) -> PolicyHost:
        return self.server.host_impl  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        """Silent by default; failures are reported by _send_json instead."""

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        if not payload.get("success", True):
            print(
                f"[insight_policy] {self.command} {self.path} -> {payload.get('error')}",
                file=sys.stderr,
            )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        if not data.strip():
            return {}
        parsed = json.loads(data.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def do_GET(self) -> None:  # name fixed by BaseHTTPRequestHandler
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/health":
            try:
                self._send_json(self._host.health())
            except Exception as exc:  # never 5xx; report the reason
                self._send_json({"healthy": False, "success": False, "error": str(exc)})
            return
        self._send_json({"success": False, "error": f"no such route: {self.path}"}, status=404)

    def do_POST(self) -> None:  # name fixed by BaseHTTPRequestHandler
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/reset":
            self._handle_reset()
        elif route == "/act":
            self._handle_act()
        elif route == "/finish":
            self._handle_finish()
        else:
            self._send_json({"success": False, "error": f"no such route: {self.path}"}, status=404)

    def _handle_reset(self) -> None:
        try:
            payload = self._read_json()
            # A first frame may be sent along; it is deliberately NOT forwarded.
            # The runner sends the same frame again as the first /act, and
            # forwarding both would duplicate frame 0 in the model's history.
            self._send_json(
                self._host.reset(
                    _text_field(payload, "instruction"),
                    episode_id=_text_field(payload, "episode_id"),
                )
            )
        except Exception as exc:  # protocol says 200 + success:false
            self._send_json({"success": False, "error": str(exc)})

    def _handle_act(self) -> None:
        frame_id = 0
        episode_id = ""
        try:
            payload = self._read_json()
            frame_id = int(_text_field(payload, "frame_id", "0") or "0")
            episode_id = _text_field(payload, "episode_id")
            jpeg = _jpeg_field(payload, "image_jpeg_b64", "image", "image_jpeg", "frame")
            rgb = decode_jpeg_b64(jpeg)
            self._send_json(self._host.act(rgb, frame_id=frame_id, episode_id=episode_id))
        except Exception as exc:  # protocol says 200 + success:false
            self._send_json(
                {
                    "success": False,
                    "episode_id": episode_id,
                    "frame_id": int(frame_id),
                    "action_name": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _handle_finish(self) -> None:
        try:
            payload = self._read_json()
            self._send_json(self._host.finish(episode_id=_text_field(payload, "episode_id")))
        except Exception as exc:  # protocol says 200 + success:false
            self._send_json({"success": False, "error": str(exc)})


def make_server(policy: Policy, *, host: str = "127.0.0.1", port: int = 18081) -> PolicyHTTPServer:
    """Bind a server for ``policy``. Port 0 binds an ephemeral port (tests)."""
    return PolicyHTTPServer((host, int(port)), _Handler, PolicyHost(policy))


def serve(policy: Policy, *, host: str = "127.0.0.1", port: int = 18081) -> None:
    """Bind and serve until interrupted.

    The policy is fully constructed by the time this is called, so a 200 from
    ``/health`` means the weights are loaded and the first ``/act`` will not
    time out behind a cold start.
    """
    httpd = make_server(policy, host=host, port=port)
    bound_host, bound_port = httpd.server_address[:2]
    print(
        f"[insight_policy] {policy.model_id} ready on http://{bound_host}:{bound_port}",
        file=sys.stderr,
    )
    # A SIGTERM is how the eval scripts stop this process, and Python's default
    # handler exits without unwinding, so neither `finally` nor atexit runs and
    # a policy holding a child process leaks it. Turning the signal into the
    # same exception a Ctrl-C raises gives the policy its close() either way.
    def _stop(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            # Not the main thread; the caller owns signal handling.
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[insight_policy] shutting down", file=sys.stderr)
    finally:
        httpd.server_close()
        try:
            policy.close()
        except Exception as exc:  # a teardown must not mask why we are here
            print(f"[insight_policy] close() raised: {exc}", file=sys.stderr)
