# Hosted SAM2 Bbox Backend

This backend exposes the small API used by the TrackMe/LabelMe GUI:

- `GET /healthz`
- `GET /readyz`
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
export SAM2_API_TOKEN='replace-with-a-long-random-secret'
# Optional resource limits (defaults shown):
export SAM2_MAX_UPLOAD_BYTES=26214400
export SAM2_MAX_PIXELS=40000000
export SAM2_CACHE_FRAMES=8
uvicorn server.sam2_backend.app:app --host 0.0.0.0 --port 9090
```

Point the GUI at the server with either `hosted_sam2.url` in `.labelmerc` or:

```bash
export LABELME_HOSTED_SAM2_URL=http://SERVER_HOST:9090
export LABELME_HOSTED_SAM2_API_TOKEN='replace-with-a-long-random-secret'
```

`/healthz` is a liveness check; `/readyz` validates model configuration and
checkpoint availability. Authentication is optional only when `SAM2_API_TOKEN`
is unset. Do not expose an unauthenticated service publicly. Terminate TLS at a
trusted reverse proxy and use an HTTPS GUI URL for any non-local deployment.

Uploads are read only up to the configured compressed-byte limit, decoded image
dimensions are capped, and prepared frames are retained in a bounded LRU cache.
