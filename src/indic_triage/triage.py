"""Severity scoring, batch analysis, clustering, and latency metrics.

Pipeline for a single session:

1. Validate + parse (``models.parse_session``) -- malformed input is reported
   as exit code 2.
2. Script/locale drift rules (``scripts.analyze_script_locale``).
3. Entity slot-corruption guard (``entities.compare_entities``).
4. Latency spikes per turn (first-token and response ms vs.
   ``max_latency_ms``).
5. Terminal outcomes (``unclear``, ``no_answer``, ``tool_error``,
   ``retry_loop``, ``hangup_early``).

Exit codes: ``0`` healthy, ``1`` any HIGH / CRITICAL failure, ``2`` invalid
or malformed input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .entities import Entities, compare_entities, extract_entities, merge_entities
from .models import (
    DEFAULT_MAX_LATENCY_MS,
    SEVERITY_RANK,
    Issue,
    Session,
    Turn,
    ValidationError,
    parse_session,
)
from .scripts import analyze_script_locale

# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

def percentile(sorted_values: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile over an ascending list."""
    if not sorted_values:
        return None
    index = max(0, math.ceil(p / 100 * len(sorted_values)) - 1)
    return sorted_values[index]


def latency_metrics(turns: Tuple[Turn, ...]) -> Dict[str, Dict[str, Optional[float]]]:
    first = sorted(t.first_token_ms for t in turns if t.first_token_ms is not None)
    response = sorted(t.response_ms for t in turns if t.response_ms is not None)
    return {
        "first_token": {"p50": percentile(first, 50), "p95": percentile(first, 95)},
        "response": {"p50": percentile(response, 50), "p95": percentile(response, 95)},
    }


def _latency_issues(turn: Turn, index: int, max_latency_ms: int) -> List[Issue]:
    issues: List[Issue] = []
    for field_name in ("first_token_ms", "response_ms"):
        value = getattr(turn, field_name)
        if value is None or value <= max_latency_ms:
            continue
        severity = "CRITICAL" if value > 3 * max_latency_ms else "HIGH"
        issues.append(
            Issue(
                severity=severity,
                category="latency_spike",
                message=(
                    f"{field_name} of {value}ms exceeds the "
                    f"{max_latency_ms}ms latency budget"
                ),
                turn_index=index,
                evidence={field_name: value, "max_latency_ms": max_latency_ms},
            )
        )
    return issues


def _entity_issues(session: Session) -> List[Issue]:
    issues: List[Issue] = []
    known: Entities = Entities()
    pending: Entities = Entities()
    for index, turn in enumerate(session.turns):
        if turn.role in ("user", "tool"):
            entities = extract_entities(turn.text)
            known = merge_entities(known, entities)
            pending = merge_entities(pending, entities)
        else:
            entities = extract_entities(turn.text)
            issues.extend(compare_entities(known, pending, entities))
            # The agent has answered; anything it did not preserve is now stale.
            pending = Entities()
    return issues


def _outcome_issues(session: Session) -> List[Issue]:
    issues: List[Issue] = []
    for outcome in session.outcomes:
        issues.append(
            Issue(
                severity="HIGH",
                category=f"outcome_{outcome}",
                message=f"Terminal outcome recorded: {outcome}",
                evidence={"outcome": outcome},
            )
        )
    return issues


def _sort_issues(issues: List[Issue]) -> List[Issue]:
    return sorted(
        issues,
        key=lambda i: (
            -SEVERITY_RANK[i.severity],
            i.turn_index if i.turn_index is not None else 10**9,
            i.category,
        ),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisResult:
    session_id: str
    verdict: str
    exit_code: int
    issues: Tuple[Issue, ...]
    latency: Dict[str, Dict[str, Optional[float]]]
    error: Optional[str] = None
    # Raw per-turn latencies, used for accurate batch aggregates (not serialized).
    first_token_raw: Tuple[float, ...] = ()
    response_raw: Tuple[float, ...] = ()

    @property
    def top_severity(self) -> Optional[str]:
        if not self.issues:
            return None
        return max((i.severity for i in self.issues), key=SEVERITY_RANK.get)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "top_severity": self.top_severity,
            "latency": self.latency,
            "issues": [i.to_dict() for i in self.issues],
            "error": self.error,
        }


@dataclass(frozen=True)
class Cluster:
    signature: str
    severity: str
    count: int
    sessions: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "severity": self.severity,
            "count": self.count,
            "sessions": list(self.sessions),
        }


@dataclass
class BatchResult:
    directory: str
    results: List[AnalysisResult] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    latency: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    @property
    def total_sessions(self) -> int:
        return len(self.results)

    @property
    def failed_sessions(self) -> int:
        return sum(1 for r in self.results if r.exit_code == 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "total_sessions": self.total_sessions,
            "failed_sessions": self.failed_sessions,
            "skipped": self.skipped,
            "clusters": [c.to_dict() for c in self.clusters],
            "latency": self.latency,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_session(data: Any, max_latency_ms: int = DEFAULT_MAX_LATENCY_MS) -> AnalysisResult:
    """Analyze a single session payload (assumed valid after parse)."""
    session = parse_session(data)
    issues: List[Issue] = []
    issues.extend(analyze_script_locale(session))
    issues.extend(_entity_issues(session))
    for index, turn in enumerate(session.turns):
        issues.extend(_latency_issues(turn, index, max_latency_ms))
    issues.extend(_outcome_issues(session))
    issues = _sort_issues(issues)

    top = issues[0].severity if issues else None
    verdict = "FAILED" if top in ("HIGH", "CRITICAL") else "HEALTHY"
    exit_code = 1 if verdict == "FAILED" else 0
    first_raw = tuple(t.first_token_ms for t in session.turns if t.first_token_ms is not None)
    response_raw = tuple(t.response_ms for t in session.turns if t.response_ms is not None)
    return AnalysisResult(
        session_id=session.session_id,
        verdict=verdict,
        exit_code=exit_code,
        issues=tuple(issues),
        latency=latency_metrics(session.turns),
        first_token_raw=first_raw,
        response_raw=response_raw,
    )


def analyze_payload(data: Any, max_latency_ms: int = DEFAULT_MAX_LATENCY_MS) -> AnalysisResult:
    """Analyze a single session payload, catching validation errors."""
    try:
        return analyze_session(data, max_latency_ms=max_latency_ms)
    except ValidationError as exc:
        return AnalysisResult(
            session_id="<malformed>",
            verdict="MALFORMED",
            exit_code=2,
            issues=(),
            latency=latency_metrics(()),
            error=str(exc),
        )


def _cluster_sessions(results: List[AnalysisResult]) -> List[Cluster]:
    groups: Dict[str, List[AnalysisResult]] = {}
    for result in results:
        if result.exit_code != 1:
            continue
        signature = "|".join(sorted({i.category for i in result.issues})) or "unknown"
        groups.setdefault(signature, []).append(result)
    clusters: List[Cluster] = []
    for signature, members in groups.items():
        severity = max((m.top_severity for m in members), key=SEVERITY_RANK.get) or "HIGH"
        clusters.append(
            Cluster(
                signature=signature,
                severity=severity,
                count=len(members),
                sessions=tuple(m.session_id for m in members),
            )
        )
    clusters.sort(key=lambda c: (-c.count, -SEVERITY_RANK[c.severity], c.signature))
    return clusters


def analyze_batch(directory: Path, max_latency_ms: int = DEFAULT_MAX_LATENCY_MS) -> BatchResult:
    """Analyze every ``*.json`` file in a directory.

    Malformed files are skipped (recorded in ``skipped``) so that valid
    sessions are still analyzed cleanly.
    """
    results: List[AnalysisResult] = []
    skipped: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            results.append(analyze_session(data, max_latency_ms=max_latency_ms))
        except (ValidationError, json.JSONDecodeError, OSError) as exc:
            skipped.append({"file": path.name, "error": str(exc)})
    return BatchResult(
        directory=str(directory),
        results=results,
        skipped=skipped,
        clusters=_cluster_sessions(results),
        latency=_aggregate_latency(results),
    )


def _aggregate_latency(results: List[AnalysisResult]) -> Dict[str, Dict[str, Optional[float]]]:
    first: List[float] = []
    response: List[float] = []
    for result in results:
        first.extend(result.first_token_raw)
        response.extend(result.response_raw)
    return {
        "first_token": {"p50": percentile(sorted(first), 50), "p95": percentile(sorted(first), 95)},
        "response": {"p50": percentile(sorted(response), 50), "p95": percentile(sorted(response), 95)},
    }
