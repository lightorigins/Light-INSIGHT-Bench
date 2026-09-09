"""Tiny localhost HTTP client used by benchmark runners."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import requests


def _as_optional_int(*candidates: Any) -> int | None:
    """The first candidate that is an integer, or ``None``."""
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_float(value: Any) -> float | None:
    """A float, or ``None`` when the server deliberately sent no number.

    ``None`` is a fact here, not a missing value: it means the step was answered
    from a queued action and the model was never run.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, *, default: int) -> int:
    """``int(value)`` when that is meaningful, else *default*. Never raises."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_xy(value: Any) -> tuple[int, int] | None:
    """Normalize a JSON ``[x, y]`` posxy pair (JSON has no tuples) to ``(int, int)``, or None."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def encode_rgb_jpeg(rgb_frame: Any, *, quality: int = 85, subsampling: int | None = None) -> bytes:
    """JPEG-encode an RGB frame exactly as the wire protocol does.

    Shared with the in-process driver so a model sees byte-identical input
    whether it is hosted over HTTP or called directly.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgb_frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected RGB frame with shape (H, W, >=3), got {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": int(quality)}
    if subsampling is not None:
        save_kwargs["subsampling"] = int(subsampling)
    Image.fromarray(arr, mode="RGB").save(buf, **save_kwargs)
    return buf.getvalue()


def _raise_for_status_with_body(response: requests.Response) -> None:
    if response.status_code < 400:
        return

    request = response.request
    method = request.method if request is not None else "HTTP"
    url = response.url or "<unknown-url>"
    body = response.text.strip()
    message = f"{method} {url} failed with HTTP {response.status_code}"
    if body:
        message = f"{message}; response body: {body}"
    import requests

    raise requests.HTTPError(message, response=response, request=request)


@dataclass(frozen=True)
class PolicyResponse:
    success: bool
    action_name: str
    raw_output: str = ""
    waypoint: dict[str, Any] | None = None
    waypoint_cluster_id: int | None = None
    waypoint_horizon: int = 0
    # Dual-pointing grounding parsed by the server; None on non-pointing ckpts. A ckpt uses one
    # scheme: grid ids (48x27) OR posxy bins (x, y each 0..999); the unused pair stays None.
    apos_id: int | None = None
    opos_id: int | None = None
    apos_xy: tuple[int, int] | None = None
    opos_xy: tuple[int, int] | None = None
    # Scheme-independent directive codes (apos 0=point/1=rot-L/2=rot-R/3=stop,
    # opos 0=point/1=not-visible). On a posxy ckpt these are the ONLY place a rot/stop frame is
    # visible: its sentinel carries neither an id nor an xy.
    apos_kind: int | None = None
    opos_kind: int | None = None
    frame_id: int = 0
    #: ``None`` on a step the server answered from a queued action rather than
    #: by running the model. Distinguishing that from a very fast inference is
    #: what makes "was the model asked on this step?" answerable at all.
    inference_time_ms: float | None = 0.0
    timings_ms: dict[str, float | None] = field(default_factory=dict)
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyResponse:
        return cls(
            success=bool(payload.get("success", False)),
            action_name=str(payload.get("action_name", "")),
            raw_output=str(payload.get("raw_output", "")),
            waypoint=payload.get("waypoint") if isinstance(payload.get("waypoint"), dict) else None,
            # Top level first, then the nested copy. Both exist because this
            # SDK's own server writes both; a server that writes only the nested
            # one used to land `null` in every step of every trace, silently.
            # Reading the fallback makes that a compatibility detail instead of
            # a rule every integrator has to remember.
            waypoint_cluster_id=_as_optional_int(
                payload.get("waypoint_cluster_id"),
                (payload.get("waypoint") or {}).get("cluster_id")
                if isinstance(payload.get("waypoint"), dict)
                else None,
            ),
            waypoint_horizon=int(payload.get("waypoint_horizon", 0) or 0),
            apos_id=(int(payload["apos_id"]) if payload.get("apos_id") is not None else None),
            opos_id=(int(payload["opos_id"]) if payload.get("opos_id") is not None else None),
            apos_xy=_as_xy(payload.get("apos_xy")),
            opos_xy=_as_xy(payload.get("opos_xy")),
            apos_kind=(int(payload["apos_kind"]) if payload.get("apos_kind") is not None else None),
            opos_kind=(int(payload["opos_kind"]) if payload.get("opos_kind") is not None else None),
            frame_id=int(payload.get("frame_id", 0) or 0),
            inference_time_ms=_optional_float(payload.get("inference_time_ms", 0.0)),
            timings_ms={str(k): float(v) for k, v in (payload.get("timings_ms") or {}).items()},
            error=str(payload.get("error", "")),
            extra=dict(payload.get("extra") or {}),
        )


class LocalVlnPolicyClient:
    """Client for the four-endpoint VLN HTTP policy protocol.

    Frames are expected as RGB numpy arrays. They are JPEG-encoded and sent as
    base64 JSON to avoid multipart dependencies in the local benchmark path.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18081",
        *,
        timeout_sec: float = 120.0,
        jpeg_quality: int = 85,
        jpeg_subsampling: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.jpeg_quality = int(jpeg_quality)
        self.jpeg_subsampling = None if jpeg_subsampling is None else int(jpeg_subsampling)
        import requests

        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        response = self._session.get(f"{self.base_url}/health", timeout=min(self.timeout_sec, 10.0))
        _raise_for_status_with_body(response)
        return dict(response.json())

    def reset(
        self,
        instruction: str,
        *,
        episode_id: str = "",
        first_frame_rgb: Any | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instruction": instruction,
            "episode_id": episode_id,
        }
        if first_frame_rgb is not None:
            encode_t0 = time.monotonic()
            payload["first_frame_jpeg_b64"] = self._encode_rgb_jpeg_b64(first_frame_rgb)
            encode_ms = (time.monotonic() - encode_t0) * 1000.0
        else:
            encode_ms = 0.0
        post_t0 = time.monotonic()
        response = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout_sec
        )
        post_ms = (time.monotonic() - post_t0) * 1000.0
        _raise_for_status_with_body(response)
        parse_t0 = time.monotonic()
        result = dict(response.json())
        parse_ms = (time.monotonic() - parse_t0) * 1000.0
        if not result.get("success", False):
            raise RuntimeError(str(result.get("error") or "VLN policy server reset failed"))
        result["client_timings_ms"] = {
            "encode_first_frame_jpeg_ms": encode_ms,
            "http_post_ms": post_ms,
            "json_parse_ms": parse_ms,
        }
        return result

    def act(
        self,
        rgb_frame: Any,
        *,
        episode_id: str = "",
        frame_id: int = 0,
    ) -> PolicyResponse:
        encode_t0 = time.monotonic()
        image_jpeg_b64 = self._encode_rgb_jpeg_b64(rgb_frame)
        encode_ms = (time.monotonic() - encode_t0) * 1000.0
        payload = {
            "episode_id": episode_id,
            "frame_id": int(frame_id),
            "timestamp_ms": int(time.time() * 1000),
            "image_jpeg_b64": image_jpeg_b64,
        }
        post_t0 = time.monotonic()
        response = self._session.post(
            f"{self.base_url}/act", json=payload, timeout=self.timeout_sec
        )
        post_ms = (time.monotonic() - post_t0) * 1000.0
        _raise_for_status_with_body(response)
        parse_t0 = time.monotonic()
        result = PolicyResponse.from_dict(dict(response.json()))
        parse_ms = (time.monotonic() - parse_t0) * 1000.0
        if not result.success:
            raise RuntimeError(result.error or "VLN policy server returned success=false")
        return replace(
            result,
            timings_ms={
                **result.timings_ms,
                "client_encode_jpeg_ms": encode_ms,
                "client_http_post_ms": post_ms,
                "client_json_parse_ms": parse_ms,
            },
        )

    def finish(self, *, episode_id: str = "") -> dict[str, Any]:
        response = self._session.post(
            f"{self.base_url}/finish",
            json={"episode_id": episode_id},
            timeout=min(self.timeout_sec, 10.0),
        )
        _raise_for_status_with_body(response)
        result = dict(response.json())
        if not result.get("success", False):
            raise RuntimeError(str(result.get("error") or "VLN policy server finish failed"))
        return result

    def close(self) -> None:
        self._session.close()

    def _encode_rgb_jpeg_b64(self, rgb_frame: Any) -> str:
        payload = encode_rgb_jpeg(
            rgb_frame, quality=self.jpeg_quality, subsampling=self.jpeg_subsampling
        )
        return base64.b64encode(payload).decode("ascii")
