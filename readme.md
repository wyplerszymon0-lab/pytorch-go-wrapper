# PyTorch ↔ Go FastAPI Wrapper

Async FastAPI service that acts as an interface layer between a Go backend and a PyTorch model.

```
Go Backend  ──►  FastAPI Wrapper  ──►  PyTorch Model
     ▲                │                      │
     └────────────────┘◄─────────────────────┘
     (preprocess / postprocess calls)
```

## File structure

```
├── main.py          # FastAPI app, routes, middleware
├── schemas.py       # Pydantic v2 request/response models
├── model.py         # PyTorch ModelManager (load, predict, unload)
├── config.py        # pydantic-settings (env / .env driven)
├── requirements.txt
└── tests/
    └── test_api.py
```

## Quick start

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration (`.env` or env vars)

| Variable | Default | Description |
|---|---|---|
| `GO_BACKEND_URL` | `http://localhost:9090` | Base URL of your Go service |
| `GO_TIMEOUT` | `10.0` | HTTP timeout (seconds) |
| `GO_PREPROCESS_ENABLED` | `false` | Call Go `/api/preprocess` before inference |
| `GO_POSTPROCESS_ENABLED` | `false` | Call Go `/api/postprocess` after inference |
| `API_KEY_REQUIRED` | `false` | Enforce `X-API-Key` header |
| `API_KEY` | `` | Expected key value |
| `MODEL_WEIGHTS_PATH` | `null` | Path to `.pt` state-dict |
| `MODEL_INPUT_DIM` | `128` | Feature vector length |
| `MODEL_NUM_CLASSES` | `10` | Output classes |
| `USE_GPU` | `false` | Use CUDA if available |
| `USE_TORCH_COMPILE` | `false` | Enable `torch.compile` (PyTorch ≥ 2.0) |
| `MAX_BATCH_SIZE` | `64` | Max items per `/predict/batch` call |
| `BATCH_CONCURRENCY` | `8` | Parallel semaphore for batch requests |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness / readiness probe |
| `POST` | `/predict` | Single-sample inference |
| `POST` | `/predict/batch` | Concurrent batch inference |
| `POST` | `/go/forward` | Thin proxy to Go backend |

Interactive docs: `http://localhost:8000/docs`

## Input types

The wrapper accepts three discriminated input types:

```jsonc
// Tabular
{ "input": { "features": [1.0, 2.0, ...] } }

// Text
{ "input": { "text": "Hello world", "language": "en" } }

// Image (base-64)
{ "input": { "data": "<base64>", "format": "jpeg" } }
```

## Go integration

When `GO_PREPROCESS_ENABLED=true`, the wrapper `POST`s to `{GO_BACKEND_URL}/api/preprocess`
before inference and expects JSON back. Similarly for `GO_POSTPROCESS_ENABLED`.
The Go service can enrich the payload with:

```json
{ "scale_factor": 1.5, "class_labels": ["cat", "dog", ...] }
```

## Adapting to your model

1. Replace `_ExampleNet` in `model.py` with your `nn.Module`.
2. Update `_preprocess()` — tokenise text, apply image transforms, etc.
3. Update `_postprocess()` — decode logits, apply threshold, etc.
4. Adjust `MODEL_INPUT_DIM` / `MODEL_NUM_CLASSES` in your `.env`.

## Testing

```bash
pytest tests/ -v --asyncio-mode=auto
```
