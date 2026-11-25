"""Typed data models and strict validation for indic-turn-triage.

All external input (session JSON) passes through :func:`parse_session`, which
raises :class:`ValidationError` with a clear message for any malformed payload:
wrong types, unknown roles, negative latencies, invalid outcomes, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VALID_ROLES = frozenset({"user", "agent", "tool"})

# Terminal outcomes that indicate a failed call, independent of content.
TERMINAL_OUTCOMES = frozenset(
    {"unclear", "no_answer", "tool_error", "retry_loop", "hangup_early"}
)

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

DEFAULT_MAX_LATENCY_MS = 1500


class ValidationError(ValueError):
    """Raised when a session payload is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


@dataclass(frozen=True)
class Turn:
    """A single conversational turn."""

    role: str
    text: str
    first_token_ms: Optional[int] = None
    response_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "first_token_ms": self.first_token_ms,
            "response_ms": self.response_ms,
        }


@dataclass(frozen=True)
class Session:
    """A validated conversation transcript."""

    session_id: str
    turns: Tuple[Turn, ...]
    language_policy: Tuple[str, ...] = ()
    outcomes: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language_policy": list(self.language_policy),
            "outcomes": list(self.outcomes),
            "meta": self.meta,
            "turns": [t.to_dict() for t in self.turns],
        }


@dataclass(frozen=True)
class Issue:
    """A deterministic finding on a turn or session."""

    severity: str
    category: str
    message: str
    turn_index: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", self.severity.upper())
        object.__setattr__(self, "category", self.category.lower())
        if self.severity not in SEVERITY_RANK:
            raise ValidationError(f"unknown severity: {self.severity!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "turn_index": self.turn_index,
            "evidence": self.evidence,
        }


def _parse_int_field(raw: Any, key: str, turn_index: int) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValidationError(f"turn[{turn_index}] {key} must be an integer, got bool")
    if isinstance(raw, int):
        if raw < 0:
            raise ValidationError(f"turn[{turn_index}] {key} must be >= 0, got {raw}")
        return raw
    if isinstance(raw, float) and raw.is_integer():
        if raw < 0:
            raise ValidationError(f"turn[{turn_index}] {key} must be >= 0, got {raw}")
        return int(raw)
    raise ValidationError(f"turn[{turn_index}] {key} must be a non-negative integer, got {raw!r}")


def parse_turn(raw: Any, turn_index: int) -> Turn:
    _require(isinstance(raw, dict), f"turn[{turn_index}] must be an object")
    role = raw.get("role")
    _require(
        role in VALID_ROLES,
        f"turn[{turn_index}] role {role!r} invalid; expected one of {sorted(VALID_ROLES)}",
    )
    text = raw.get("text")
    _require(isinstance(text, str), f"turn[{turn_index}] text must be a string")
    first_token_ms = _parse_int_field(raw.get("first_token_ms"), "first_token_ms", turn_index)
    response_ms = _parse_int_field(raw.get("response_ms"), "response_ms", turn_index)
    return Turn(role=role, text=text, first_token_ms=first_token_ms, response_ms=response_ms)


def _parse_outcome(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    values: List[str]
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list) and all(isinstance(v, str) for v in raw):
        values = list(raw)
    else:
        raise ValidationError(
            "outcome must be a string or a list of strings from "
            f"{sorted(TERMINAL_OUTCOMES)}"
        )
    for v in values:
        _require(
            v in TERMINAL_OUTCOMES,
            f"outcome {v!r} invalid; expected one of {sorted(TERMINAL_OUTCOMES)}",
        )
    return tuple(values)


def parse_session(data: Any) -> Session:
    """Validate and normalize a session payload.

    Raises :class:`ValidationError` with a precise message on malformed input.
    """
    _require(isinstance(data, dict), "session payload must be a JSON object")
    session_id = data.get("session_id")
    _require(isinstance(session_id, str) and session_id.strip(), "missing string field 'session_id'")

    raw_turns = data.get("turns")
    _require(isinstance(raw_turns, list) and raw_turns, "'turns' must be a non-empty list")
    turns = tuple(parse_turn(t, i) for i, t in enumerate(raw_turns))

    policy_raw = data.get("language_policy", [])
    if policy_raw is None:
        policy_raw = []
    _require(isinstance(policy_raw, list), "'language_policy' must be a list of strings")
    policy = tuple(str(p) for p in policy_raw if isinstance(p, str) and str(p).strip())

    outcomes = _parse_outcome(data.get("outcome"))

    meta_raw = data.get("meta", {})
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    return Session(
        session_id=session_id,
        turns=turns,
        language_policy=policy,
        outcomes=outcomes,
        meta=dict(meta_raw),
    )
