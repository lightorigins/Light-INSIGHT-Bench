from __future__ import annotations

import pytest
import requests

from insight_bench.vln_runtime.policy.client import _raise_for_status_with_body


def test_raise_for_status_with_body_includes_fastapi_detail() -> None:
    request = requests.Request("POST", "http://127.0.0.1:18081/reset").prepare()
    response = requests.Response()
    response.status_code = 422
    response.url = request.url
    response.request = request
    response._content = b'{"detail":[{"loc":["body","instruction"],"msg":"Field required"}]}'

    with pytest.raises(requests.HTTPError) as exc_info:
        _raise_for_status_with_body(response)

    message = str(exc_info.value)
    assert "POST http://127.0.0.1:18081/reset failed with HTTP 422" in message
    assert "instruction" in message
    assert "Field required" in message
