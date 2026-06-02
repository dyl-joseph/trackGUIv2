import os
import posixpath
from urllib.parse import urlparse
from urllib.parse import urlunparse

import requests


class HostedSam2Error(RuntimeError):
    pass


class HostedSam2Client:
    def __init__(
        self,
        url=None,
        api_token=None,
        timeout_seconds=30,
        verify_tls=True,
    ):
        self.url = (url or "").rstrip("/")
        self.api_token = api_token or None
        self.timeout_seconds = timeout_seconds
        self.verify_tls = verify_tls

    @classmethod
    def from_config(cls, config):
        hosted_config = config.get("hosted_sam2", {})
        url = hosted_config.get("url") or os.environ.get("LABELME_HOSTED_SAM2_URL")
        api_token = hosted_config.get("api_token") or os.environ.get(
            "LABELME_HOSTED_SAM2_API_TOKEN"
        )
        return cls(
            url=url,
            api_token=api_token,
            timeout_seconds=hosted_config.get("timeout_seconds", 30),
            verify_tls=hosted_config.get("verify_tls", True),
        )

    def is_configured(self):
        return bool(self.url)

    def register_image(self, image_data, client_frame_key=None):
        if not self.is_configured():
            raise HostedSam2Error("Hosted SAM2 URL is not configured.")
        data = {}
        if client_frame_key:
            data["client_frame_key"] = client_frame_key
        response = self._post(
            "/v1/images",
            files={"image": ("frame.jpg", image_data, "application/octet-stream")},
            data=data,
        )
        image_id = response.get("image_id")
        width = response.get("width")
        height = response.get("height")
        if not image_id or not isinstance(width, int) or not isinstance(height, int):
            raise HostedSam2Error("Hosted SAM2 returned an invalid image response.")
        return response

    def point_prompt(self, image_id, x, y, label=1):
        if not self.is_configured():
            raise HostedSam2Error("Hosted SAM2 URL is not configured.")
        response = self._post(
            "/v1/point-prompts",
            json={
                "image_id": image_id,
                "x": float(x),
                "y": float(y),
                "label": int(label),
            },
        )
        bbox = response.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise HostedSam2Error("Hosted SAM2 returned an invalid bbox response.")
        return response

    def _post(self, path, **kwargs):
        try:
            response = requests.post(
                self._url(path),
                headers=self._headers(),
                timeout=self.timeout_seconds,
                verify=self.verify_tls,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise HostedSam2Error(str(exc)) from exc
        return self._decode_response(response)

    def _decode_response(self, response):
        try:
            payload = response.json()
        except ValueError as exc:
            raise HostedSam2Error(
                "Hosted SAM2 returned a non-JSON response "
                f"(HTTP {response.status_code})."
            ) from exc
        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise HostedSam2Error(
                detail or f"Hosted SAM2 failed ({response.status_code})."
            )
        if not isinstance(payload, dict):
            raise HostedSam2Error("Hosted SAM2 returned an invalid JSON payload.")
        return payload

    def _headers(self):
        if not self.api_token:
            return {}
        return {"Authorization": f"Bearer {self.api_token}"}

    def _url(self, path):
        parsed = urlparse(self.url)
        joined_path = posixpath.join(parsed.path.rstrip("/"), path.lstrip("/"))
        return urlunparse(parsed._replace(path=joined_path))
