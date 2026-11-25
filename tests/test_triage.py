"""Tests for the triage scorer, batch clustering, and latency metrics."""

import json
from pathlib import Path

from indic_triage.triage import analyze_batch, analyze_payload, percentile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_percentile():
    assert percentile([100, 200, 300, 400, 500], 50) == 300
    assert percentile([100, 200, 300, 400, 500], 95) == 500
    assert percentile([100, 200, 300, 400, 500], 0) == 100
    assert percentile([], 50) is None


def test_healthy_session_exits_zero():
    result = analyze_payload(load("healthy_hinglish_session.json"))
    assert result.exit_code == 0
    assert result.verdict == "HEALTHY"
    assert result.issues == ()
    assert result.latency["first_token"]["p50"] is not None


def test_failing_session_one_exits_one():
    result = analyze_payload(load("failing_session_1.json"))
    assert result.exit_code == 1
    assert result.verdict == "FAILED"
    assert result.top_severity == "CRITICAL"
    categories = {i.category for i in result.issues}
    assert "entity_altered" in categories
    assert "entity_dropped" in categories
    assert "latency_spike" in categories
    assert "script_flip" in categories
    assert "garbled_mixed_script" in categories
    assert "locale_mismatch" in categories
    assert "outcome_retry_loop" in categories
    spike = next(i for i in result.issues if i.category == "latency_spike")
    assert spike.severity == "HIGH"
    assert spike.evidence["response_ms"] == 3200


def test_failing_session_two_exits_one():
    result = analyze_payload(load("failing_session_2.json"))
    assert result.exit_code == 1
    categories = {i.category for i in result.issues}
    assert "locale_mismatch" in categories
    assert "outcome_tool_error" in categories
    assert "outcome_hangup_early" in categories


def test_malformed_session_exits_two():
    result = analyze_payload(load("malformed_session.json"))
    assert result.exit_code == 2
    assert result.verdict == "MALFORMED"
    assert result.error


def test_batch_clustering():
    batch = analyze_batch(FIXTURES)
    assert batch.total_sessions == 3
    assert batch.failed_sessions == 2
    assert len(batch.skipped) == 1
    assert batch.skipped[0]["file"] == "malformed_session.json"
    signatures = {c.signature for c in batch.clusters}
    assert len(signatures) == 2
    assert all(c.count == 1 for c in batch.clusters)
    assert all(c.severity in ("HIGH", "CRITICAL") for c in batch.clusters)


def test_batch_aggregate_latency():
    batch = analyze_batch(FIXTURES)
    first = batch.latency["first_token"]
    assert first["p50"] is not None and first["p95"] is not None
    assert first["p95"] >= first["p50"]


def test_max_latency_threshold_scoring():
    session = load("healthy_hinglish_session.json")
    # A tiny budget turns every agent latency into a spike.
    result = analyze_payload(session, max_latency_ms=1)
    assert result.exit_code == 1
    assert any(i.category == "latency_spike" for i in result.issues)
