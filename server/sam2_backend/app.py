import json
import math
import os

from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Header
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import field_validator
from starlette.concurrency import run_in_threadpool

from .auth import bearer_token_is_valid
from .service import Sam2Service
from .service import Sam2ServiceError

app = FastAPI(title="Hosted SAM2 Bbox Backend")
service = Sam2Service.from_env()
api_token = os.environ.get("SAM2_API_TOKEN")
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class RequestBodyTooLarge(Exception):
    pass


class IngressGuardMiddleware:
    """Authenticate and cap request bodies before Starlette parses multipart."""

    protected_paths = {"/v1/images", "/v1/point-prompts"}

    def __init__(
        self,
        app,
        max_body_bytes_getter,
        token_getter,
        multipart_overhead_bytes=MAX_MULTIPART_OVERHEAD_BYTES,
    ):
        self.app = app
        self.max_body_bytes_getter = max_body_bytes_getter
        self.token_getter = token_getter
        self.multipart_overhead_bytes = int(multipart_overhead_bytes)

    @staticmethod
    async def _send_error(send, status_code, detail):
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        path = (scope.get("path") or "").rstrip("/") or "/"
        if scope.get("type") != "http" or path not in self.protected_paths:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode(
            "latin-1", errors="replace"
        )
        if not bearer_token_is_valid(self.token_getter(), authorization):
            await self._send_error(send, 401, "Invalid bearer token.")
            return

        max_body_bytes = (
            int(self.max_body_bytes_getter()) + self.multipart_overhead_bytes
        )
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = None
            if declared_length is not None and declared_length > max_body_bytes:
                await self._send_error(send, 413, "Request body exceeds ingress limit.")
                return

        received_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > max_body_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._send_error(send, 413, "Request body exceeds ingress limit.")


app.add_middleware(
    IngressGuardMiddleware,
    max_body_bytes_getter=lambda: service.max_upload_bytes,
    token_getter=lambda: api_token,
)


class PointPromptRequest(BaseModel):
    image_id: str
    x: float
    y: float
    label: int = 1

    @field_validator("image_id", mode="before")
    def validate_image_id(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("image_id must be a non-empty string")
        return value

    @field_validator("x", "y", mode="before")
    def validate_coordinate(cls, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("point coordinates must be finite numbers")
        return value

    @field_validator("label", mode="before")
    def validate_label(cls, value):
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError("point label must be 0 or 1")
        return value


def authorize(authorization: str = Header(None)):
    if not bearer_token_is_valid(api_token, authorization):
        raise HTTPException(status_code=401, detail="Invalid bearer token.")


@app.get("/healthz")
def healthz():
    return service.health()


@app.get("/readyz")
def readyz():
    readiness = service.readiness()
    if not readiness["ready"]:
        raise HTTPException(status_code=503, detail=readiness["detail"])
    return readiness


@app.post("/v1/images")
async def register_image(
    image: UploadFile = File(...),
    client_frame_key: str = Form(None),
    _authorized=Depends(authorize),
):
    try:
        image_bytes = await image.read(service.max_upload_bytes + 1)
        return await run_in_threadpool(
            service.register_image,
            image_bytes,
            client_frame_key=client_frame_key,
        )
    except Sam2ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/v1/point-prompts", dependencies=[Depends(authorize)])
def point_prompt(payload: PointPromptRequest):
    try:
        return service.point_prompt(
            image_id=payload.image_id,
            x=payload.x,
            y=payload.y,
            label=payload.label,
        )
    except Sam2ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
