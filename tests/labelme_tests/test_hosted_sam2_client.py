import math

import pytest

from labelme.hosted_sam2_client import HostedSam2Client
from labelme.hosted_sam2_client import HostedSam2Error


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_http_error_retains_status_for_cache_recovery():
    client = HostedSam2Client("https://sam.example")

    with pytest.raises(HostedSam2Error) as exc_info:
        client._decode_response(FakeResponse(404, {"detail": "Unknown image_id."}))

    assert exc_info.value.status_code == 404
    assert exc_info.value.payload == {"detail": "Unknown image_id."}


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, True, 4],
        [0, 0, math.inf, 4],
        [4, 0, 2, 4],
        [0, 4, 2, 2],
    ],
)
def test_point_prompt_rejects_invalid_bbox_values(monkeypatch, bbox):
    client = HostedSam2Client("https://sam.example")
    monkeypatch.setattr(
        client,
        "_post",
        lambda *_args, **_kwargs: {"bbox": bbox, "score": 0.5},
    )

    with pytest.raises(HostedSam2Error, match="invalid bbox"):
        client.point_prompt("image", 1, 1)


def test_point_prompt_rejects_boolean_and_nonfinite_coordinates():
    client = HostedSam2Client("https://sam.example")

    with pytest.raises(HostedSam2Error, match="finite numbers"):
        client.point_prompt("image", True, 1)
    with pytest.raises(HostedSam2Error, match="finite numbers"):
        client.point_prompt("image", math.nan, 1)

    with pytest.raises(HostedSam2Error, match="Point label"):
        client.point_prompt("image", 1, 1, label=True)
