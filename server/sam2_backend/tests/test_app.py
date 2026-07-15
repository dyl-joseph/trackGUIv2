import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
HTTPException = fastapi.HTTPException
app_module = pytest.importorskip("server.sam2_backend.app")


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
