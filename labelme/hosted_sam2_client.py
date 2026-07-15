import math
import os
import posixpath
from urllib.parse import urlparse
from urllib.parse import urlunparse

import requests


class HostedSam2Error(RuntimeError):
    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


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
        self._session = requests.Session()

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
        if (
            not isinstance(image_id, str)
            or not image_id.strip()
            or isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise HostedSam2Error("Hosted SAM2 returned an invalid image response.")
        return response

    def point_prompt(self, image_id, x, y, label=1):
        if not self.is_configured():
            raise HostedSam2Error("Hosted SAM2 URL is not configured.")
        if not isinstance(image_id, str) or not image_id.strip():
            raise HostedSam2Error("image_id must be a non-empty string.")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
        ):
            raise HostedSam2Error("Point coordinates must be finite numbers.")
        if isinstance(label, bool) or label not in (0, 1):
            raise HostedSam2Error("Point label must be 0 or 1.")
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
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in bbox
            )
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise HostedSam2Error("Hosted SAM2 returned an invalid bbox response.")
        score = response.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise HostedSam2Error("Hosted SAM2 returned an invalid score.")
        return response

    def _post(self, path, **kwargs):
        try:
            response = self._session.post(
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
                f"(HTTP {response.status_code}).",
                status_code=response.status_code,
            ) from exc
        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise HostedSam2Error(
                detail or f"Hosted SAM2 failed ({response.status_code}).",
                status_code=response.status_code,
                payload=payload,
            )
        if not isinstance(payload, dict):
            raise HostedSam2Error("Hosted SAM2 returned an invalid JSON payload.")
        return payload

    def close(self):
        self._session.close()

    def _headers(self):
        if not self.api_token:
            return {}
        return {"Authorization": f"Bearer {self.api_token}"}

    def _url(self, path):
        parsed = urlparse(self.url)
        joined_path = posixpath.join(parsed.path.rstrip("/"), path.lstrip("/"))
        return urlunparse(parsed._replace(path=joined_path))
