"""Policy wire protocol: the HTTP client a run uses to call your model.

Only the run side lives here. The model side of the wire is a separate program
in a separate Python environment -- see ``policies/`` in this repository -- and
it never imports this package, which is why nothing here reaches towards it.

The names below are resolved on first use (PEP 562) rather than at import time,
so importing the package costs nothing until a run actually opens the wire.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .client import LocalVlnPolicyClient, PolicyResponse, encode_rgb_jpeg

__all__ = ["LocalVlnPolicyClient", "PolicyResponse", "encode_rgb_jpeg"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
