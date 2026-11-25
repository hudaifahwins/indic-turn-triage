"""Unicode script detection, romanized-Indic lexical markers, and entity guards."""
from __future__ import annotations

import re
import unicodedata

from .models import Script, Severity, TriageIssue, Turn

# Unicode script ranges (scanned per codepoint via unicodedata name fallback).
DEVANAGARI_RANGE = (0x0900, 0x097F)
TAMIL_RANGE = (0x0B00, 0x0B7F)
ARABIC_RANGE = (0x0600, 0x06FF)
CYRILLIC_RANGE = (0x0400, 0x04FF)
CJK_RANGE = [
    (0x1100, 0x11FF), (0x2E80, 0x2EFF), (0x3000, 0x30FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
]

ROMANIZED_MARKERS: frozenset[str] = frozenset(
    {
        "kal", "baje", "paisa", "rupees", "anna", "nahi", "haan", "konjam",
        "seri", "achha", "bura", "jaldi", "bhai", "didi", "pila", "paise",
        "tumse", "tere", "yahan", "wahan", "abhi", "ab", "ha", "hain", "hai",
    }
)

LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’\-]*")
DEVANAGARI_WORD_RE = re.compile(r"[\u0900-\u097F]+")
TAMIL_WORD_RE = re.compile(r"[\u0B00-\u0B7F]+")
ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF]+")
CJK_WORD_RE = re.compile(r"[\u1100-\u11FF\u2E80-\u30FF\u3400-\u9FFF\uF900-\uFAFF]+")


def char_script(ch: str) -> Script:
    code = ord(ch)
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
            return Script.DEVANAGARI
        if TAMIL_RANGE[0] <= code <= TAMIL_RANGE[1]:
            return Script.TAMIL
        if ARABIC_RANGE[0] <= code <= ARABIC_RANGE[1]:
            return Script.ARABIC
        if CYRILLIC_RANGE[0] <= code <= CYRILLIC_RANGE[1]:
            return Script.LATIN
        if any(a <= code <= b for a, b in CJK_RANGE):
            return Script.CJK
        return Script.LATIN
    if "Nd" in cat or "Nl" in cat or "No" in cat:
        return Script.LATIN
    if cat.startswith("P"):
        return Script.UNKNOWN
    return Script.OTHER


def split_tokens(text: str) -> list[str]:
    """Split into whitespace-free script-aware tokens (mixed-script kept whole)."""
    tokens: list[str] = []
    buf: list[str] = []
    current_script: Script | None = None
    for ch in text:
        s = char_script(ch)
        if buf and s != current_script:
            tokens.append("".join(buf))
            buf = []
        buf.append(ch)
        current_script = s
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t.strip()]


def token_script(token: str) -> Script:
    scripts = {char_script(c) for c in token}
    scripts.discard(Script.UNKNOWN)
    scripts.discard(Script.OTHER)
    if not scripts:
        return Script.UNKNOWN
    if len(scripts) == 1:
        return next(iter(scripts))
    # mixed-script token (e.g. Devanagari prefix + Latin suffix) -> garbled
    return Script.LATIN  # caller detects mixing via token script set


def detect_scripts(text: str) -> list[Script]:
    seen: list[Script] = []
    for token in split_tokens(text):
        s = token_script(token)
        if s not in seen:
            seen.append(s)
    return seen


def has_script_flip(text: str) -> bool:
    tokens = split_tokens(text)
    if len(tokens) < 2:
        return False
    scripts = [token_script(t) for t in tokens]
    pairs = list(zip(scripts, scripts[1:]))
    for a, b in pairs:
        if a in (Script.LATIN, Script.DEVANAGARI, Script.TAMIL) and b in (
            Script.LATIN, Script.DEVANAGARI, Script.TAMIL,
        ) and a is not b:
            return True
    return False


def has_garbled_mixed_script(text: str) -> list[str]:
    bad: list[str] = []
    for token in split_tokens(text):
        s = token_script(token)
        if s is Script.LATIN and any(
            char_script(c) in (Script.DEVANAGARI, Script.TAMIL, Script.ARABIC, Script.CJK)
            for c in token
        ):
            bad.append(token)
    return bad


def detect_romanized_markers(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for marker in ROMANIZED_MARKERS:
        if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lowered):
            found.append(marker)
    return found


CURRENCY_RE = re.compile(
    r"(?:₹|INR|Rs\.?|Rs|INR[₹])?\s*([\d][\d,]*(?:\.\d+)?)\s*"
    r"(?:lakh|lakh\s+crore|crore|thousand|k|thousand\s+units)?",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"\+?\d{1,2}[-\s]?\d{9,12}|\b\d{10}\b")
BOOKING_RE = re.compile(r"\b[A-Z]{2,4}-\d{3,}\b|\bBK[0-9]{3,}\b|\b[A-Z0-9]{6,12}-[0-9]{2,}\b")


def extract_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for m in CURRENCY_RE.finditer(text):
        digits = m.group(1).replace(",", "").replace(".", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        unit = (m.group(2) or "unit").lower()
        multiplier = 1.0
        if unit.startswith("lakh"):
            multiplier = 100_000.0
        elif unit.startswith("crore") or unit.startswith("cr"):
            multiplier = 10_000_000.0
        elif unit.startswith("k"):
            multiplier = 1_000.0
        amounts.append(round(value * multiplier, 2))
    return amounts


def extract_phones(text: str) -> list[str]:
    return [m.group(0).replace(" ", "") for m in PHONE_RE.finditer(text)]


def extract_booking_ids(text: str) -> list[str]:
    return [m.group(0) for m in BOOKING_RE.finditer(text)]


def make_issue(code: str, category: str, severity: str, turn_id: int | None, detail: str) -> TriageIssue:
    return TriageIssue(code=code, category=category, severity=severity, turn_id=turn_id, detail=detail)
