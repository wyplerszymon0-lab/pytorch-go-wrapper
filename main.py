import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from config import Settings
from model import ModelManager
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

settings = Settings()
model_manager = ModelManager(settings)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client

    logger.info("⚡ Starting up — loading model …")
    await asyncio.to_thread(model_manager.load)
    logger.info("✅ Model ready")

    http_client = httpx.AsyncClient(
        base_url=settings.go_backend_url,
        timeout=settings.go_timeout,
        headers={"X-Service-Name": "pytorch-wrapper"},
    )
    logger.info("🔗 HTTP client connected to Go backend: %s", settings.go_backend_url)

    yield

    logger.info("🛑 Shutting down …")
    await http_client.aclose()
    model_manager.unload()
    logger.info("👋 Clean shutdown complete")


app = FastAPI(
    title="PyTorch ↔ Go API Wrapper",
    description="Async FastAPI service bridging a Go backend and a PyTorch model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%-6s %s  →  %d  (%.1f ms)  [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error [%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred.",
            request_id=request_id,
        ).model_dump(),
    )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)):
    if not settings.api_key_required:
        return
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def require_model() -> ModelManager:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded. Please retry shortly.",
        )
    return model_manager


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness / readiness probe",
)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        model_name=settings.model_name,
        go_backend_url=settings.go_backend_url,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Single-sample inference",
    dependencies=[Depends(verify_api_key)],
)
async def predict(
    body: PredictRequest,
    request: Request,
    mgr: ModelManager = Depends(require_model),
):
    request_id = request.state.request_id

    go_context: dict[str, Any] = {}
    if settings.go_preprocess_enabled:
        go_context = await _call_go(
            "/api/preprocess",
            {"input": body.input, "request_id": request_id},
        )

    raw_output = await asyncio.to_thread(mgr.predict, body.input, go_context)

    if settings.go_postprocess_enabled:
        raw_output = await _call_go(
            "/api/postprocess",
            {"output": raw_output, "request_id": request_id},
        )

    return PredictResponse(
        request_id=request_id,
        predictions=raw_output["predictions"],
        confidence=raw_output.get("confidence"),
        metadata=raw_output.get("metadata", {}),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    summary="Batch inference (concurrent)",
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch(
    body: BatchPredictRequest,
    request: Request,
    mgr: ModelManager = Depends(require_model),
):
    request_id = request.state.request_id

    if len(body.inputs) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {len(body.inputs)} exceeds limit of {settings.max_batch_size}.",
        )

    sem = asyncio.Semaphore(settings.batch_concurrency)

    async def _infer_one(idx: int, sample):
        async with sem:
            result = await asyncio.to_thread(mgr.predict, sample, {})
            return idx, result

    tasks = [_infer_one(i, inp) for i, inp in enumerate(body.inputs)]
    indexed_results = await asyncio.gather(*tasks, return_exceptions=True)

    results, errors = [], []
    for item in indexed_results:
        if isinstance(item, Exception):
            errors.append(str(item))
        else:
            idx, r = item
            results.append(
                PredictResponse(
                    request_id=f"{request_id}:{idx}",
                    predictions=r["predictions"],
                    confidence=r.get("confidence"),
                    metadata=r.get("metadata", {}),
                )
            )

    return BatchPredictResponse(request_id=request_id, results=results, errors=errors)


@app.post(
    "/go/forward",
    tags=["proxy"],
    summary="Forward an arbitrary payload to the Go backend",
    dependencies=[Depends(verify_api_key)],
)
async def forward_to_go(payload: dict[str, Any], request: Request):
    return await _call_go("/api/forward", payload)


async def _call_go(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if http_client is None:
        raise RuntimeError("HTTP client not initialised.")
    try:
        response = await http_client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Go backend returned %d for %s", exc.response.status_code, path)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Go backend error: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Could not reach Go backend at %s: %s", path, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Go backend is unreachable.",
        ) from exc
