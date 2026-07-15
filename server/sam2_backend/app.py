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

from .auth import bearer_token_is_valid
from .service import Sam2Service
from .service import Sam2ServiceError

app = FastAPI(title="Hosted SAM2 Bbox Backend")
service = Sam2Service.from_env()
api_token = os.environ.get("SAM2_API_TOKEN")


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
        return service.register_image(
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
