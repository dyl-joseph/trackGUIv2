from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel

from .service import Sam2Service
from .service import Sam2ServiceError

app = FastAPI(title="Hosted SAM2 Bbox Backend")
service = Sam2Service.from_env()


class PointPromptRequest(BaseModel):
    image_id: str
    x: float
    y: float
    label: int = 1


@app.get("/healthz")
def healthz():
    return service.health()


@app.post("/v1/images")
async def register_image(
    image: UploadFile = File(...),
    client_frame_key: str = Form(None),
):
    try:
        return service.register_image(
            await image.read(),
            client_frame_key=client_frame_key,
        )
    except Sam2ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/v1/point-prompts")
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
