import asyncio
import threading

import httpx
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


def _run_asgi(app, scope, request_messages, receive_error=None):
    sent = []
    messages = list(request_messages)

    async def receive():
        if receive_error is not None:
            raise receive_error
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def test_ingress_guard_rejects_unauthenticated_request_before_reading_body():
    async def unreachable_app(_scope, _receive, _send):
        raise AssertionError("inner app must not run")

    middleware = app_module.IngressGuardMiddleware(
        unreachable_app,
        max_body_bytes_getter=lambda: 2,
        token_getter=lambda: "secret-token",
        multipart_overhead_bytes=0,
    )
    scope = {"type": "http", "path": "/v1/images", "headers": []}

    sent = _run_asgi(
        middleware,
        scope,
        [],
        receive_error=AssertionError("request body must not be read"),
    )

    assert sent[0]["status"] == 401


def test_ingress_guard_caps_chunked_body_before_multipart_parsing():
    async def consuming_app(_scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = app_module.IngressGuardMiddleware(
        consuming_app,
        max_body_bytes_getter=lambda: 3,
        token_getter=lambda: "secret-token",
        multipart_overhead_bytes=0,
    )
    scope = {
        "type": "http",
        "path": "/v1/images",
        "headers": [(b"authorization", b"Bearer secret-token")],
    }
    messages = [
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"34", "more_body": False},
    ]

    sent = _run_asgi(middleware, scope, messages)

    assert sent[0]["status"] == 413


def test_health_remains_responsive_during_blocked_image_registration(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    watchdog_fired = threading.Event()

    class BlockingService:
        max_upload_bytes = 1024

        def register_image(self, _image_bytes, client_frame_key=None):
            started.set()
            release.wait(timeout=2)
            return {
                "image_id": "registered",
                "width": 1,
                "height": 1,
                "prepared": True,
            }

        def health(self):
            return {"status": "ok", "model": "test", "cached_images": 0}

    monkeypatch.setattr(app_module, "api_token", None)
    monkeypatch.setattr(app_module, "service", BlockingService())

    def release_on_deadlock():
        watchdog_fired.set()
        release.set()

    async def exercise_app():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            registration = asyncio.create_task(
                client.post(
                    "/v1/images",
                    files={"image": ("frame.png", b"x", "image/png")},
                )
            )
            while not started.is_set():
                await asyncio.sleep(0.01)
            health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
            release.set()
            registered = await registration
            return health, registered

    watchdog = threading.Timer(1, release_on_deadlock)
    watchdog.start()
    try:
        health, registered = asyncio.run(exercise_app())
    finally:
        release.set()
        watchdog.cancel()

    assert not watchdog_fired.is_set()
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert registered.status_code == 200
