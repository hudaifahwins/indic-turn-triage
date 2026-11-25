"""Unicode script analysis and romanized Indic heuristics.

Pure-Python, dependency-free script detection:

* ``LATIN``         -- Basic Latin / Latin-1 / Latin Extended (letters)
* ``DEVANAGARI``    -- U+0900..U+097F (Devanagari, used for Hindi)
* ``TAMIL``         -- U+0B80..U+0BFF (Tamil script)
* ``DIGITS_PUNCT``  -- digits, punctuation, symbols, separators (incl. ``₹``)
* ``OTHER`` / ``MIXED`` -- fallbacks

Romanized Hinglish and Tanglish are recognized through lexical markers
(e.g. ``kal``, ``baje``, ``paisa``, ``rupees``, ``anna``, ``nahi``,
``haan`` for Hinglish; ``konjam``, ``seri`` and friends for Tanglish).

The module also implements the deterministic code-switch / locale-drift
rules used by the triage scorer.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

from .models import Issue, SEVERITY_RANK

# ---------------------------------------------------------------------------
# Unicode script detection
# ---------------------------------------------------------------------------

_LATIN_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin-1 supplement + extended A/B
    (0x1E00, 0x1EFF),  # Latin extended additional
    (0x2C60, 0x2C7F),  # Latin extended C
)
_DEVANAGARI = (0x0900, 0x097F)
_TAMIL = (0x0B80, 0x0BFF)

LETTER_SCRIPTS = ("LATIN", "DEVANAGARI", "TAMIL")


def script_of(char: str) -> str:
    """Classify a single character into a script bucket."""
    code = ord(char)
    for lo, hi in _LATIN_RANGES:
        if lo <= code <= hi:
            return "LATIN"
    if _DEVANAGARI[0] <= code <= _DEVANAGARI[1]:
        return "DEVANAGARI"
    if _TAMIL[0] <= code <= _TAMIL[1]:
        return "TAMIL"
    category = unicodedata.category(char)
    if category.startswith(("Nd", "P", "S", "Z")):
        return "DIGITS_PUNCT"
    return "OTHER"


def classify_text(text: str) -> Counter:
    """Count characters per script bucket for a whole string."""
    counts: Counter = Counter()
    for ch in text:
        counts[script_of(ch)] += 1
    return counts


def token_scripts(text: str) -> List[Tuple[str, str]]:
    """Return ``(token, script)`` pairs; a token is ``MIXED`` when it mixes
    letters from more than one writing system (Devanagari+Tamil, Latin+Tamil,
    or Latin+Devanagari inside the same word)."""
    out: List[Tuple[str, str]] = []
    for token in re.findall(r"\S+", text):
        scripts: set[str] = set()
        for ch in token:
            scripts.add(script_of(ch))
        letters = {s for s in scripts if s in LETTER_SCRIPTS}
        if len(letters) > 1:
            out.append((token, "MIXED"))
        elif len(letters) == 1:
            out.append((token, letters.pop()))
        elif scripts - {"DIGITS_PUNCT"}:
            out.append((token, "OTHER"))
        else:
            out.append((token, "DIGITS_PUNCT"))
    return out


def garbled_tokens(text: str) -> List[str]:
    """Tokens that mix scripts inside a single word (garbled mixed-script)."""
    return [tok for tok, scr in token_scripts(text) if scr == "MIXED"]


# ---------------------------------------------------------------------------
# Romanized Hinglish / Tanglish lexical markers
# ---------------------------------------------------------------------------

HINGLISH_MARKERS = frozenset(
    {
        # Spec-nominated markers
        "kal", "baje", "paisa", "rupees", "anna", "nahi", "haan",
        # Extended common Hinglish vocabulary
        "paisay", "paise", "rupaye", "nahin", "hain", "hai", "kya", "ho",
        "hum", "main", "tum", "aap", "theek", "thik", "achha", "achcha",
        "sahi", "samajh", "baat", "kaam", "kar", "karo", "karta", "karte",
        "gaya", "gaye", "gya", "chahiye", "bolo", "bol", "suno", "dekho",
        "namaste", "dhanyavad", "shukriya", "madad", "batana", "batao",
        "bata", "kharcha", "bharosa", "zaroor", "bilkul", "thoda", "bahut",
        "wala", "wali", "waali", "log", "sab", "kuch", "kyun", "kyon",
        "kaise", "acha", "suna", "suni", "aapka", "aapki", "mera", "meri",
        "tera", "teri", "kyaa", "hoga", "hogaya", "ji", "sirji", "bhaiya",
        "didiji", "sarkar", "paan", "tiffin", "jugaad", "kamaal", "accha",
    }
)

TANGLISH_MARKERS = frozenset(
    {
        # Spec-nominated markers
        "konjam", "seri",
        # Extended common Tanglish vocabulary
        "romba", "nalla", "epdi", "eppadi", "enna", "eppo", "illa", "illai",
        "ille", "ippa", "inga", "anga", "vara", "vaanga", "vanga", "po",
        "poda", "sollu", "sol", "irukku", "iruku", "periya", "chinna",
        "apram", "apdi", "vesham", "mokka", "suthama", "madam", "paathu",
        "paatu", "vandhu", "poga", "pogum", "kudukka", "edukka", "venum",
        "venam", "sari", "unga", "enakku", "unakku", "avan", "aval",
        "ivanga", "vanakkam", "pannunga", "pannu", "podunga", "va", "da",
        "la", "ke", "pala", "kandha", "mudiyum", "mudiyadhu", "aagum",
        "aagadhu", "irundha", "irundhu", "sonna", "sonnen", "paathu",
        "paarthen", "vandha", "vandhu", "poga", "pogalaam", "okayva",
    }
)

_WORD_RE = re.compile(r"[a-z']+")


def marker_hits(text: str) -> Tuple[int, int]:
    """Return ``(hinglish_hits, tanglish_hits)`` for a text."""
    words = set(_WORD_RE.findall(text.lower()))
    hinglish = len(words & HINGLISH_MARKERS)
    tanglish = len(words & TANGLISH_MARKERS)
    return hinglish, tanglish


def has_hinglish(text: str) -> bool:
    return marker_hits(text)[0] > 0


def has_tanglish(text: str) -> bool:
    return marker_hits(text)[1] > 0


# ---------------------------------------------------------------------------
# Language inference
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Infer the language of a turn.

    Returns one of ``en``, ``hi``, ``ta`` or ``unknown``.

    * Literal Devanagari / Tamil script wins.
    * Otherwise, romanized Hinglish vs Tanglish markers decide.
    * Otherwise, Latin text without Indic markers is ``en``.
    """
    counts = classify_text(text)
    devanagari = counts.get("DEVANAGARI", 0)
    tamil = counts.get("TAMIL", 0)
    latin = counts.get("LATIN", 0)
    if devanagari and devanagari >= latin and devanagari >= tamil:
        return "hi"
    if tamil and tamil >= latin and tamil >= devanagari:
        return "ta"
    hinglish, tanglish = marker_hits(text)
    if tanglish > hinglish and tanglish > 0:
        return "ta"
    if hinglish > 0:
        return "hi"
    if latin > 0:
        return "en"
    return "unknown"


def _script_language(literal_script: str) -> str:
    return {
        "devanagari": "hi",
        "tamil": "ta",
        "latin": "en",
    }.get(literal_script, "unknown")


# ---------------------------------------------------------------------------
# Drift rules
# ---------------------------------------------------------------------------

def analyze_script_locale(session: "Session") -> List[Issue]:
    """Deterministic code-switch / script-drift / locale-mismatch rules.

    Flags:

    * ``script_flip``          -- a user turn suddenly switches literal script
      (e.g. romanized Hinglish -> Devanagari) vs. earlier user turns.
    * ``locale_mismatch``      -- the agent replies in a language different
      from the user's language (e.g. pure English to a Hindi/Tamil user),
      which violates bilingual support policy.
    * ``garbled_mixed_script`` -- a token mixing letters of two scripts.
    """
    issues: List[Issue] = []
    policy = set(session.language_policy)
    prev_user_literal: str | None = None
    prev_user_lang: str | None = None

    for index, turn in enumerate(session.turns):
        text = turn.text
        lang = detect_language(text)
        counts = classify_text(text)
        devanagari = counts.get("DEVANAGARI", 0)
        tamil = counts.get("TAMIL", 0)
        latin = counts.get("LATIN", 0)
        literal = (
            "devanagari"
            if devanagari > 0
            else "tamil"
            if tamil > 0
            else "latin"
            if latin > 0
            else None
        )

        # Garbled mixed-script tokens anywhere.
        garbled = garbled_tokens(text)
        if garbled:
            issues.append(
                Issue(
                    severity="MEDIUM",
                    category="garbled_mixed_script",
                    message=f"Garbled mixed-script token(s): {', '.join(garbled)}",
                    turn_index=index,
                    evidence={"tokens": garbled},
                )
            )

        if turn.role == "user":
            if (
                prev_user_literal
                and literal
                and literal != prev_user_literal
                and literal in ("devanagari", "tamil")
            ):
                script_lang = _script_language(literal)
                severity = (
                    "HIGH"
                    if policy and script_lang not in policy
                    else "MEDIUM"
                )
                issues.append(
                    Issue(
                        severity=severity,
                        category="script_flip",
                        message=(
                            f"Unexpected script flip in user turn: "
                            f"{prev_user_literal} -> {literal} ({script_lang})"
                        ),
                        turn_index=index,
                        evidence={"from": prev_user_literal, "to": literal},
                    )
                )
            prev_user_literal = literal
            prev_user_lang = lang
        elif turn.role == "agent":
            if prev_user_lang in ("hi", "ta") and lang in ("hi", "ta", "en"):
                if lang != prev_user_lang:
                    issues.append(
                        Issue(
                            severity="HIGH",
                            category="locale_mismatch",
                            message=(
                                f"Agent replied in {lang!r} while the user was "
                                f"speaking {prev_user_lang!r} (policy violation)"
                            ),
                            turn_index=index,
                            evidence={"agent_language": lang, "user_language": prev_user_lang},
                        )
                    )
    issues.sort(key=lambda i: (-SEVERITY_RANK[i.severity], i.turn_index if i.turn_index is not None else 10**9, i.category))
    return issues
