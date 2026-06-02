# Hosted SAM2 Bbox Backend

This backend exposes the small API used by the TrackMe/LabelMe GUI:

- `GET /healthz`
- `POST /v1/images`
- `POST /v1/point-prompts`

Install SAM2 on the GPU server following Meta's SAM2 instructions, then install
the lightweight web dependencies:

```bash
pip install -r server/sam2_backend/requirements.txt
pip install -e /path/to/sam2
```

Set the SAM2 model config and checkpoint before starting:

```bash
export SAM2_MODEL_CFG=/path/to/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
export SAM2_CHECKPOINT=/path/to/checkpoints/sam2.1_hiera_large.pt
export SAM2_DEVICE=cuda
uvicorn server.sam2_backend.app:app --host 0.0.0.0 --port 9090
```

Point the GUI at the server with either `hosted_sam2.url` in `.labelmerc` or:

```bash
export LABELME_HOSTED_SAM2_URL=http://SERVER_HOST:9090
```
