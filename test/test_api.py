from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

with (
    patch("model.ModelManager.load", return_value=None),
    patch("model.ModelManager.is_loaded", new_callable=lambda: property(lambda self: True)),
    patch("model.ModelManager.predict", return_value={
        "predictions": [0.1] * 10,
        "confidence": 0.9,
        "metadata": {"top_class": 0},
    }),
):
    from main import app


TABULAR_PAYLOAD = {"input": {"features": [float(i) for i in range(128)]}}
TEXT_PAYLOAD = {"input": {"text": "Hello, world!", "language": "en"}}
IMAGE_PAYLOAD = {"input": {"data": "aGVsbG8=", "format": "jpeg"}}

MOCK_OUTPUT = {"predictions": [0.1] * 10, "confidence": 0.9, "metadata": {}}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_tabular(client):
    with patch("model.ModelManager.predict", return_value=MOCK_OUTPUT):
        r = client.post("/predict", json=TABULAR_PAYLOAD)
    assert r.status_code == 200
    assert "predictions" in r.json()
    assert "request_id" in r.json()


def test_predict_text(client):
    with patch("model.ModelManager.predict", return_value=MOCK_OUTPUT):
        r = client.post("/predict", json=TEXT_PAYLOAD)
    assert r.status_code == 200


def test_predict_image(client):
    with patch("model.ModelManager.predict", return_value=MOCK_OUTPUT):
        r = client.post("/predict", json=IMAGE_PAYLOAD)
    assert r.status_code == 200


def test_predict_invalid_input(client):
    r = client.post("/predict", json={"input": {"unknown_field": 42}})
    assert r.status_code == 422


def test_batch_predict(client):
    batch = {"inputs": [TABULAR_PAYLOAD["input"]] * 3}
    with patch("model.ModelManager.predict", return_value=MOCK_OUTPUT):
        r = client.post("/predict/batch", json=batch)
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert body["errors"] == []


def test_batch_exceeds_limit(client):
    from config import Settings
    limit = Settings().max_batch_size
    oversized = {"inputs": [TABULAR_PAYLOAD["input"]] * (limit + 1)}
    r = client.post("/predict/batch", json=oversized)
    assert r.status_code == 422


def test_request_id_header_returned(client):
    r = client.get("/health", headers={"X-Request-ID": "test-abc-123"})
    assert r.headers.get("X-Request-ID") == "test-abc-123"


@pytest.mark.asyncio
async def test_predict_async():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        with patch("model.ModelManager.predict", return_value=MOCK_OUTPUT):
            r = await ac.post("/predict", json=TABULAR_PAYLOAD)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_async():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
