"""FastAPI application for the indic-turn-triage dashboard.

Endpoints:

* ``GET /``           -- interactive triage dashboard (HTML/JS)
* ``GET /health``     -- liveness probe
* ``GET /api/results``-- preloaded fixtures analysis (recomputed on demand)
* ``POST /triage``    -- analyze a single session or a batch payload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .triage import analyze_batch, analyze_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
DASHBOARD_HTML = Path(__file__).resolve().parent / "static" / "dashboard.html"

app = FastAPI(title="indic-turn-triage", version="0.1.0")

_BATCH_CACHE: Dict[str, Any] = {}


def _batch_results() -> Dict[str, Any]:
    """Analyze the fixtures directory (cached, recomputable)."""
    if "results" not in _BATCH_CACHE:
        if FIXTURES_DIR.is_dir():
            _BATCH_CACHE["results"] = analyze_batch(FIXTURES_DIR).to_dict()
        else:
            _BATCH_CACHE["results"] = {
                "directory": str(FIXTURES_DIR),
                "total_sessions": 0,
                "failed_sessions": 0,
                "skipped": [],
                "clusters": [],
                "latency": {
                    "first_token": {"p50": None, "p95": None},
                    "response": {"p50": None, "p95": None},
                },
                "results": [],
            }
    return _BATCH_CACHE["results"]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "indic-turn-triage"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))


@app.get("/api/results")
def api_results() -> Dict[str, Any]:
    return _batch_results()


def _parse_triage_body(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    if "sessions" in body:
        sessions = body["sessions"]
        if not isinstance(sessions, list):
            raise HTTPException(status_code=400, detail="'sessions' must be a list")
        results = []
        skipped = []
        for payload in sessions:
            outcome = analyze_payload(payload)
            if outcome.exit_code == 2:
                skipped.append({"error": outcome.error})
            else:
                results.append(outcome)
        return {
            "mode": "batch",
            "total_sessions": len(results),
            "failed_sessions": sum(1 for r in results if r.exit_code == 1),
            "skipped": skipped,
            "results": [r.to_dict() for r in results],
        }
    outcome = analyze_payload(body)
    if outcome.exit_code == 2:
        raise HTTPException(status_code=400, detail=outcome.error)
    return {"mode": "session", "result": outcome.to_dict()}


@app.post("/triage")
async def triage(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    return _parse_triage_body(body)
