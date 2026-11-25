"""Entity extraction and slot-corruption guard.

Extracts *critical* conversational entities from turn text:

* INR / rupee amounts (``₹``, ``Rs``, ``INR``, ``rupees``) including Indian
  style thousands separators and **lakh / crore** scale words.
* Phone numbers (10-digit Indian mobile numbers, optional ``+91`` prefix).
* Times (``HH:MM am/pm``, ``5 pm``, ``7 baje``).
* Alphanumeric booking / reference IDs (e.g. ``BK-82413``, ``TX-55120``).

:func:`compare_entities` then verifies that entities established in user/tool
turns are preserved in the agent's reply: dropped, hallucinated, or altered
values are reported as ``CRITICAL`` findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import Issue

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Amount:
    value: float
    raw: str


@dataclass(frozen=True)
class Entities:
    amounts: Tuple[Amount, ...] = ()
    phones: Tuple[str, ...] = ()
    times: Tuple[Tuple[int, int], ...] = ()
    ids: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "amounts": [{"value": a.value, "raw": a.raw} for a in self.amounts],
            "phones": list(self.phones),
            "times": [f"{h:02d}:{m:02d}" for h, m in self.times],
            "ids": list(self.ids),
        }


_AMOUNT_RE = re.compile(
    r"(?P<cur>₹|Rs\.?|rs\.?|INR|rupees?|rupaye)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>crore|crs?|cr|lakh|lacs?|lac)?",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?([6-9][\d\s-]{8,13})(?!\d)")

_TIME_HM_RE = re.compile(r"(?<![:\d])(\d{1,2}):(\d{2})\s*(am|pm)?", re.IGNORECASE)
_TIME_AP_RE = re.compile(r"(?<![:\d])(\d{1,2})\s*(am|pm)\b", re.IGNORECASE)
_TIME_BAJE_RE = re.compile(r"(?<![:\d])(\d{1,2})\s*baje\b", re.IGNORECASE)

_ID_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{3,15}\b")

_AMOUNT_SCALES = {
    "crore": 10_000_000,
    "cr": 10_000_000,
    "crs": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
}


def _to_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _parse_amounts(text: str) -> List[Amount]:
    amounts: List[Amount] = []
    for match in _AMOUNT_RE.finditer(text):
        currency = match.group("cur")
        scale_word = match.group("scale")
        # Bare numbers without a currency marker or lakh/crore scale are not
        # treated as monetary entities (avoids false positives).
        if currency is None and scale_word is None:
            continue
        base = _to_number(match.group("num"))
        if scale_word is not None:
            value = float(base * _AMOUNT_SCALES[scale_word.lower()])
        else:
            value = float(base)
        amounts.append(Amount(value=value, raw=match.group(0)))
    return amounts


def _normalize_time(hour: str, minute: str, ampm: Optional[str]) -> Tuple[int, int]:
    h = int(hour)
    m = int(minute)
    ap = (ampm or "").lower()
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h, m


def _parse_times(text: str) -> List[Tuple[int, int]]:
    times: List[Tuple[int, int]] = []
    for match in _TIME_HM_RE.finditer(text):
        h, m = _normalize_time(match.group(1), match.group(2), match.group(3))
        times.append((h, m))
    for match in _TIME_AP_RE.finditer(text):
        h, m = _normalize_time(match.group(1), "0", match.group(2))
        times.append((h, m))
    for match in _TIME_BAJE_RE.finditer(text):
        h, m = _normalize_time(match.group(1), "0", None)
        times.append((h, m))
    # De-duplicate while preserving order.
    seen: set[Tuple[int, int]] = set()
    unique: List[Tuple[int, int]] = []
    for t in times:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _parse_phones(text: str) -> List[str]:
    phones: List[str] = []
    for match in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(1))
        if len(digits) == 10 and digits[0] in "6789":
            phones.append(digits)
    return phones


def _parse_ids(text: str) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for token in _ID_RE.findall(text):
        if not any(ch.isdigit() for ch in token):
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        upper = token.upper()
        if upper in seen:
            continue
        seen.add(upper)
        ids.append(upper)
    return ids


def extract_entities(text: str) -> Entities:
    """Extract critical entities from a turn."""
    return Entities(
        amounts=tuple(_parse_amounts(text)),
        phones=tuple(_parse_phones(text)),
        times=tuple(_parse_times(text)),
        ids=tuple(_parse_ids(text)),
    )


def merge_entities(left: Entities, right: Entities) -> Entities:
    return Entities(
        amounts=left.amounts + right.amounts,
        phones=left.phones + right.phones,
        times=left.times + right.times,
        ids=left.ids + right.ids,
    )


# ---------------------------------------------------------------------------
# Comparison / slot-corruption guard
# ---------------------------------------------------------------------------

def _check_category(
    label: str,
    pending_values: List[Any],
    known_values: List[Any],
    reply_values: List[Any],
    normalize,
) -> List[Issue]:
    issues: List[Issue] = []
    pending_set = {normalize(v) for v in pending_values}
    known_set = {normalize(v) for v in known_values}
    reply_set = {normalize(v) for v in reply_values}

    if pending_values:
        if not reply_values:
            issues.append(
                Issue(
                    severity="CRITICAL",
                    category="entity_dropped",
                    message=f"{label} from user/tool turns is missing in the agent reply",
                    evidence={"expected": list(pending_values)},
                )
            )
            return issues
        unmatched_reply = [v for v in reply_values if normalize(v) not in pending_set]
        unmatched_pending = [v for v in pending_values if normalize(v) not in reply_set]
        if unmatched_pending and unmatched_reply:
            issues.append(
                Issue(
                    severity="CRITICAL",
                    category="entity_altered",
                    message=f"{label} in the agent reply does not match what user/tool turns established",
                    evidence={"expected": unmatched_pending, "got": unmatched_reply},
                )
            )
        elif unmatched_reply:
            issues.append(
                Issue(
                    severity="CRITICAL",
                    category="entity_hallucinated",
                    message=f"{label} in the agent reply is not grounded in user/tool turns",
                    evidence={"got": unmatched_reply},
                )
            )
    else:
        extras = [v for v in reply_values if normalize(v) not in known_set]
        if extras:
            issues.append(
                Issue(
                    severity="CRITICAL",
                    category="entity_hallucinated",
                    message=f"{label} in the agent reply is not grounded in user/tool turns",
                    evidence={"got": extras},
                )
            )
    return issues


def compare_entities(known: Entities, pending: Entities, reply: Entities) -> List[Issue]:
    """Compare an agent reply against entities established by user/tool turns.

    ``pending`` are entities from user/tool turns *since the last agent reply*
    (they must be preserved). ``known`` is the cumulative set used to catch
    hallucinated values.
    """
    issues: List[Issue] = []
    issues.extend(
        _check_category(
            "Currency amount",
            [a.value for a in pending.amounts],
            [a.value for a in known.amounts],
            [a.value for a in reply.amounts],
            lambda v: v,
        )
    )
    issues.extend(
        _check_category(
            "Phone number",
            list(pending.phones),
            list(known.phones),
            list(reply.phones),
            lambda v: re.sub(r"\D", "", v),
        )
    )
    issues.extend(
        _check_category(
            "Time",
            list(pending.times),
            list(known.times),
            list(reply.times),
            lambda v: v,
        )
    )
    issues.extend(
        _check_category(
            "Booking/reference ID",
            list(pending.ids),
            list(known.ids),
            list(reply.ids),
            lambda v: v.upper(),
        )
    )
    return issues
