"""Tests for entity extraction and the slot-corruption guard."""

from indic_triage.entities import Entities, compare_entities, extract_entities


def test_extract_amounts():
    entities = extract_entities("Pay ₹1,50,000 and Rs 5 lakh and 2 crore please")
    values = sorted(a.value for a in entities.amounts)
    assert values == [150000.0, 500000.0, 20000000.0]


def test_extract_amount_rupees_words():
    entities = extract_entities("The amount is rupees 340 and INR 1,200")
    values = sorted(a.value for a in entities.amounts)
    assert values == [340.0, 1200.0]


def test_bare_numbers_not_amounts():
    entities = extract_entities("Order 500 units at price 25 each")
    assert entities.amounts == ()


def test_extract_phones():
    entities = extract_entities("Call me at 9876543210 or +91 98450 12345")
    assert set(entities.phones) == {"9876543210", "9845012345"}


def test_extract_times():
    entities = extract_entities("meet at 10:30 AM, or 5 pm, or 7 baje")
    assert set(entities.times) == {(10, 30), (17, 0), (7, 0)}


def test_extract_ids():
    entities = extract_entities("Reference BK-82413 and TX-55120 please")
    assert set(entities.ids) == {"BK-82413", "TX-55120"}


def test_extract_no_spurious_ids():
    entities = extract_entities("2026-12-05 and 5 pm and ₹2,340")
    assert entities.ids == ()


def test_compare_match():
    known = extract_entities("pay ₹2,340 at 5 pm for BK-82413")
    pending = extract_entities("pay ₹2,340 at 5 pm for BK-82413")
    reply = extract_entities("payment of ₹2,340 at 5 pm confirmed for BK-82413")
    assert compare_entities(known, pending, reply) == []


def test_compare_dropped():
    known = extract_entities("total is ₹5,000")
    pending = extract_entities("total is ₹5,000")
    reply = extract_entities("your total looks fine")
    issues = compare_entities(known, pending, reply)
    assert any(i.category == "entity_dropped" for i in issues)
    assert all(i.severity == "CRITICAL" for i in issues)


def test_compare_altered():
    pending = extract_entities("balance is ₹5,000")
    reply = extract_entities("balance is ₹8,000")
    issues = compare_entities(pending, pending, reply)
    assert any(i.category == "entity_altered" for i in issues)


def test_compare_hallucinated():
    known = extract_entities("contact 9876543210")
    reply = extract_entities("call 9998887776")
    issues = compare_entities(known, Entities(), reply)
    assert any(i.category == "entity_hallucinated" for i in issues)


def test_compare_lakh_crore_equivalence():
    known = extract_entities("due amount is 5 lakh")
    pending = extract_entities("due amount is 5 lakh")
    reply = extract_entities("your due amount is ₹5,00,000")
    assert compare_entities(known, pending, reply) == []


def test_compare_phone_dropped():
    pending = extract_entities("my number is 9876543210")
    reply = extract_entities("OK, noted.")
    issues = compare_entities(pending, pending, reply)
    assert any(i.category == "entity_dropped" and "Phone" in i.message for i in issues)


def test_compare_time_altered():
    pending = extract_entities("call at 5 pm")
    reply = extract_entities("call at 6 pm")
    issues = compare_entities(pending, pending, reply)
    assert any(i.category == "entity_altered" and "Time" in i.message for i in issues)
