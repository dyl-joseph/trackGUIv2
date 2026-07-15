import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.sam2_backend import app as app_module
from server.sam2_backend.service import Sam2Service


def test_authorization_is_optional_only_without_configured_token(monkeypatch):
    monkeypatch.setattr(app_module, "api_token", None)
    assert app_module.authorize(None) is None

    monkeypatch.setattr(app_module, "api_token", "secret-token")
    with pytest.raises(HTTPException) as missing:
        app_module.authorize(None)
    with pytest.raises(HTTPException) as wrong:
        app_module.authorize("Bearer wrong")

    assert missing.value.status_code == 401
    assert wrong.value.status_code == 401
    assert app_module.authorize("Bearer secret-token") is None


def test_image_route_rejects_missing_bearer_token(monkeypatch):
    monkeypatch.setattr(app_module, "api_token", "secret-token")
    client = TestClient(app_module.app)

    response = client.post(
        "/v1/images",
        files={"image": ("frame.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid bearer token."}


def test_image_route_enforces_upload_limit(monkeypatch):
    monkeypatch.setattr(app_module, "api_token", "secret-token")
    monkeypatch.setattr(app_module, "service", Sam2Service(max_upload_bytes=2))
    client = TestClient(app_module.app)

    response = client.post(
        "/v1/images",
        headers={"Authorization": "Bearer secret-token"},
        files={"image": ("frame.png", b"123", "image/png")},
    )

    assert response.status_code == 413
    assert "2 byte limit" in response.json()["detail"]
