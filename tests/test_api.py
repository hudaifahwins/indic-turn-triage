"""Tests for the FastAPI dashboard and triage endpoints."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from indic_triage.api import app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
client = TestClient(app)


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "html" in response.text.lower()
    assert "indic-turn-triage" in response.text


def test_api_results_preloaded():
    response = client.get("/api/results")
    assert response.status_code == 200
    data = response.json()
    assert data["total_sessions"] == 3
    assert data["failed_sessions"] == 2
    assert isinstance(data["clusters"], list)
    assert "latency" in data
    assert len(data["results"]) == 3


def test_post_triage_healthy_session():
    payload = load("healthy_hinglish_session.json")
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "session"
    assert data["result"]["verdict"] == "HEALTHY"
    assert data["result"]["exit_code"] == 0


def test_post_triage_failing_session():
    payload = load("failing_session_1.json")
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    assert response.json()["result"]["exit_code"] == 1


def test_post_triage_malformed_returns_400():
    payload = load("malformed_session.json")
    response = client.post("/triage", json=payload)
    assert response.status_code == 400


def test_post_triage_batch():
    payload = {
        "sessions": [
            load("healthy_hinglish_session.json"),
            load("failing_session_1.json"),
        ]
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "batch"
    assert data["total_sessions"] == 2
    assert data["failed_sessions"] == 1


def test_post_triage_invalid_json_returns_400():
    response = client.post(
        "/triage", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400


def test_post_triage_batch_all_healthy():
    payload = {"sessions": [load("healthy_hinglish_session.json")]}
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["failed_sessions"] == 0


def test_api_results_clusters_have_fields():
    response = client.get("/api/results")
    data = response.json()
    for c in data["clusters"]:
        assert set(["signature", "severity", "count", "sessions"]).issubset(c.keys())
        assert c["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert isinstance(c["count"], int)


def test_session_result_has_exit_code_and_latency():
    response = client.post("/triage", json=load("healthy_hinglish_session.json"))
    data = response.json()["result"]
    assert "exit_code" in data
    assert "latency" in data
    assert "first_token" in data["latency"]
    assert "response" in data["latency"]
