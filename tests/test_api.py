"""Web API contract: validation, no-session behaviour, static serving.

These tests exercise the HTTP layer without a model or camera — the
heavy path (start a real session) is covered by the benchmark scripts,
which run the full pipeline on real footage.
"""

import pytest
from fastapi.testclient import TestClient

import trafficlens.web.server as server
from trafficlens.classes import COCO_CLASSES


@pytest.fixture()
def client():
    server._session = None  # isolate: no leftover session between tests
    with TestClient(server.app) as c:
        yield c
    server._session = None


def test_index_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "TrafficLens" in res.text


def test_meta_lists_models_and_classes(client):
    data = client.get("/api/meta").json()
    assert "yolo11n.pt" in data["models"]
    assert data["classes"] == COCO_CLASSES
    assert "car" in data["traffic_classes"]


def test_session_status_when_idle(client):
    assert client.get("/api/session").json() == {"running": False}


def test_start_with_missing_source_fails_cleanly(client):
    res = client.post("/api/session", json={"source": "no/such/file.mp4"})
    assert res.status_code == 400
    assert "not found" in res.json()["detail"]


def test_start_with_invalid_confidence_rejected(client):
    res = client.post("/api/session", json={"source": "x.mp4", "confidence": 2.0})
    assert res.status_code == 422


def test_start_with_bad_gate_rejected(client):
    res = client.post("/api/session", json={
        "source": "x.mp4",
        "gates": [{"name": "g", "start": [383, 297], "end": [666, 297]}],  # pixels
    })
    assert res.status_code == 422


def test_endpoints_409_without_session(client):
    for method, path in [
        ("get", "/api/events"),
        ("get", "/api/export/events.csv"),
        ("get", "/api/export/summary.json"),
        ("get", "/api/violations"),
        ("delete", "/api/session"),
        ("get", "/api/snapshot.jpg"),
    ]:
        res = getattr(client, method)(path)
        assert res.status_code == 409, f"{method} {path} -> {res.status_code}"


def test_put_gates_without_session_409(client):
    res = client.put("/api/gates", json={"gates": []})
    assert res.status_code == 409


def test_bad_speed_unit_rejected(client):
    res = client.put("/api/speed", json={"unit": "furlongs"})
    assert res.status_code == 422
