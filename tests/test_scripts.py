"""Tests for Unicode script analysis and romanized Indic heuristics."""

from indic_triage.models import Session, Turn
from indic_triage.scripts import (
    analyze_script_locale,
    classify_text,
    detect_language,
    garbled_tokens,
    has_hinglish,
    has_tanglish,
    script_of,
    token_scripts,
)


def test_script_of():
    assert script_of("a") == "LATIN"
    assert script_of("A") == "LATIN"
    assert script_of("é") == "LATIN"
    assert script_of("अ") == "DEVANAGARI"
    assert script_of("क") == "DEVANAGARI"
    assert script_of("அ") == "TAMIL"
    assert script_of("5") == "DIGITS_PUNCT"
    assert script_of("₹") == "DIGITS_PUNCT"
    assert script_of(" ") == "DIGITS_PUNCT"
    assert script_of("!") == "DIGITS_PUNCT"


def test_classify_text():
    counts = classify_text("अभी 123!")
    assert counts["DEVANAGARI"] >= 1
    assert counts["DIGITS_PUNCT"] >= 4
    counts = classify_text("vanakkam 123")
    assert counts["LATIN"] >= 6


def test_token_scripts():
    tokens = dict(token_scripts("नमस्तेhello 123 ₹500"))
    assert tokens["नमस्तेhello"] == "MIXED"
    assert tokens["123"] == "DIGITS_PUNCT"
    assert tokens["₹500"] == "DIGITS_PUNCT"


def test_garbled_tokens():
    assert garbled_tokens("नमस्तेhello") == ["नमस्तेhello"]
    assert garbled_tokens("plain english") == []
    assert garbled_tokens("अच्छा") == []


def test_marker_detection():
    assert has_hinglish("kal paisa baje nahi haan")
    assert has_tanglish("konjam seri")
    assert has_hinglish("rupees anna")
    assert not has_hinglish("completely plain English sentence")
    assert not has_tanglish("completely plain English sentence")


def test_detect_language():
    assert detect_language("Hello, how are you?") == "en"
    assert detect_language("कल पैसा दो") == "hi"
    assert detect_language("अभी मिलते हैं") == "hi"
    assert detect_language("kal baje paisa nahi haan") == "hi"
    assert detect_language("konjam seri illa") == "ta"
    assert detect_language("Vanakkam! Enakku konjam help venum") == "ta"
    assert detect_language("?????") == "unknown"


def _session(turns):
    return Session(
        session_id="t",
        turns=tuple(Turn(role=r, text=t) for r, t in turns),
        language_policy=("hi",),
    )


def test_script_flip_flag():
    session = _session(
        [
            ("user", "kal main bill check karna hai"),
            ("user", "कल मैं बिल देखना चाहता हूँ"),
        ]
    )
    issues = analyze_script_locale(session)
    flips = [i for i in issues if i.category == "script_flip"]
    assert len(flips) == 1
    assert flips[0].severity == "MEDIUM"  # Hindi is in policy -> MEDIUM


def test_script_flip_high_when_not_in_policy():
    session = Session(
        session_id="t",
        turns=(
            Turn(role="user", text="kal main bill check karna hai"),
            Turn(role="user", text="அது எனக்கு பிடிக்கும்"),
        ),
        language_policy=("hi",),
    )
    issues = analyze_script_locale(session)
    flips = [i for i in issues if i.category == "script_flip"]
    assert flips and flips[0].severity == "HIGH"


def test_locale_mismatch_flag():
    session = Session(
        session_id="t",
        turns=(
            Turn(role="user", text="konjam seri illa"),
            Turn(role="agent", text="Sure, I will take care of it right now."),
        ),
    )
    issues = analyze_script_locale(session)
    mismatches = [i for i in issues if i.category == "locale_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].severity == "HIGH"


def test_no_mismatch_when_agent_uses_same_language():
    session = Session(
        session_id="t",
        turns=(
            Turn(role="user", text="kal paisa bhejo"),
            Turn(role="agent", text="Haan, main abhi bhejta hoon."),
        ),
    )
    issues = analyze_script_locale(session)
    assert not [i for i in issues if i.category == "locale_mismatch"]


def test_garbled_flag_any_role():
    session = Session(
        session_id="t",
        turns=(Turn(role="user", text="नमस्तेhello kya hua"),),
    )
    issues = analyze_script_locale(session)
    garbled = [i for i in issues if i.category == "garbled_mixed_script"]
    assert len(garbled) == 1
    assert "नमस्तेhello" in garbled[0].message
