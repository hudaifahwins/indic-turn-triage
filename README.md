# indic-turn-triage

Deterministic triage for **real multilingual Indian voice/chat agent failures**.
It analyzes conversation-turn transcripts across English, Hindi/Hinglish, and
Tamil-English (Tanglish) to catch the failure classes that actually break
bilingual voice agents:

1. **Code-switch, script drift & romanized Indic heuristics** — pure-Python
   Unicode script detection (`LATIN`, `DEVANAGARI`, `TAMIL`,
   `DIGITS_PUNCT`) plus romanized Hinglish/Tanglish lexical markers
   (`kal`, `baje`, `paisa`, `rupees`, `anna`, `nahi`, `haan`,
   `konjam`, `seri`, …) that flag unexpected script flips, locale
   mismatches (agent replying in pure English when the user switched to
   Hindi/Tamil against bilingual policy), and garbled mixed-script tokens.
2. **Entity, amount & slot corruption guard** — verifies that critical
   entities mentioned in user/tool turns (`₹`/INR amounts, `lakh`/`crore`
   values, phone numbers, dates/times, booking/reference IDs) are not
   dropped, hallucinated, or altered in the agent's reply.
3. **Latency, outcome scoring, batch clustering & localhost web UI** —
   deterministic per-turn severity scoring (`CRITICAL`, `HIGH`, `MEDIUM`,
   `LOW`) across script/locale drift, entity corruption, latency spikes
   (first-token & response ms), and terminal outcomes (`unclear`,
   `no_answer`, `tool_error`, `retry_loop`, `hangup_early`). Supports
   single-session analysis, batch directory clustering, p50/p95 latency
   metrics, and a FastAPI + HTML/JS triage dashboard.

---

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install fastapi uvicorn pytest httpx
```

## Run the tests

```bash
.venv/bin/pytest -v
```

## CLI usage

```bash
# Single session (exit 0 = healthy, 1 = failures, 2 = malformed input)
.venv/bin/indic-triage --input fixtures/healthy_hinglish_session.json
.venv/bin/indic-triage --input fixtures/failing_session_1.json
.venv/bin/indic-triage --input fixtures/malformed_session.json

# Batch directory (valid sessions analyzed cleanly; malformed files skipped)
.venv/bin/indic-triage --input-dir fixtures

# Options
.venv/bin/indic-triage --input fixtures/failing_session_1.json --max-latency-ms 900
.venv/bin/indic-triage --input fixtures/healthy_hinglish_session.json --format json
.venv/bin/indic-triage --input-dir fixtures --format markdown --output report.md
```

## Localhost web dashboard

```bash
nohup .venv/bin/uvicorn indic_triage.api:app --host 127.0.0.1 --port 8002 > server.log 2>&1 &
```

Then open **http://localhost:8002** (or `curl http://127.0.0.1:8002/health`
for a liveness check). The dashboard is preloaded with the `fixtures/`
analysis and shows clustered failure patterns, p50/p95 latency, ranked broken
turns, and a live `POST /triage` tester.

### Sample output

```text
======================================================================
indic-turn-triage analysis
======================================================================
Session    : failing_session_1
Verdict   : FAILED (exit 1)
Latency   : first-token p50=140ms p95=190ms | response p50=700ms p95=3200ms
Issues    : 9

  [CRITICAL] entity_altered @ session: Currency amount in the agent reply does not match what user/tool turns established
  [CRITICAL] entity_dropped @ session: Phone number from user/tool turns is missing in the agent reply
  [CRITICAL] entity_dropped @ session: Phone number from user/tool turns is missing in the agent reply
  [HIGH] locale_mismatch @ turn #3: Agent replied in 'en' while the user was speaking 'hi' (policy violation)
  [HIGH] latency_spike @ turn #5: response_ms of 3200ms exceeds the 1500ms latency budget
  [HIGH] outcome_retry_loop @ session: Terminal outcome recorded: retry_loop
  [MEDIUM] garbled_mixed_script @ turn #2: Garbled mixed-script token(s): नमस्तेhello,
  [MEDIUM] script_flip @ turn #2: Unexpected script flip in user turn: latin -> devanagari (hi)
```

---

## Design Choices

### 1. Deterministic Unicode / romanized-Indic drift

- **Pure-Python Unicode script detection** — no external NLP libraries. Each
  character is bucketed by Unicode code point ranges + `unicodedata`
  categories into `LATIN`, `DEVANAGARI` (U+0900–097F), `TAMIL`
  (U+0B80–0BFF), and `DIGITS_PUNCT`. A token mixing letters from two
  scripts is flagged as `garbled_mixed_script`.
- **Romanized heuristics** — language inference for romanized text uses
  curated lexical marker sets (Hinglish: `kal`, `baje`, `paisa`,
  `rupees`, `anna`, `nahi`, `haan`, …; Tanglish: `konjam`, `seri`, …).
  Script evidence wins, markers break ties, plain Latin defaults to English.
- **Drift rules** — `script_flip` fires when a user turn suddenly moves into
  a native script (romanized → Devanagari/Tamil) or between scripts, with
  severity raised when the target language is outside the session's
  `language_policy`. `locale_mismatch` fires when the agent replies in a
  different language from the user's current language (e.g. pure English to
  a Hindi/Tamil user) — the exact bilingual-policy violation.

### 2. Entity-slot preservation

- The extractor recognizes INR amounts (symbols and words) with Indian
  separators and `lakh`/`crore` scale words normalized to canonical numeric
  values, 10-digit phone numbers (optional `+91`), times (`HH:MM am/pm`,
  `5 pm`, `7 baje`), and alphanumeric booking/reference IDs.
- The comparator tracks entities established in user/tool turns. A reply
  that omits them is `entity_dropped`; a value differing from the
  established one is `entity_altered`; a value with no grounding in
  user/tool turns is `entity_hallucinated` — all `CRITICAL`.

### 3. Batch failure clustering & deterministic scoring

- Each issue carries a `(severity, category, turn, message)` tuple;
  sessions are scored with a fixed rule order (drift → entities → latency →
  outcomes) so results are fully reproducible.
- Batch mode clusters failing sessions by their **root-cause signature**
  (sorted issue-category set), giving operators a ranked view of recurring
  failure patterns. Latency percentile (p50/p95) metrics are computed with a
  nearest-rank percentile over raw per-turn values.
- Exit codes are machine-friendly: `0` healthy, `1` HIGH/CRITICAL failures,
  `2` invalid/malformed input. Malformed files in a batch directory are
  skipped with a warning so the remaining valid sessions are still analyzed.

---

## Repository layout

```
indic-turn-triage/
├── pyproject.toml
├── src/indic_triage/
│   ├── models.py      # typed dataclasses + strict validation
│   ├── scripts.py     # Unicode script analyzer + romanized Indic markers + drift rules
│   ├── entities.py    # entity extractor & slot-corruption comparator
│   ├── triage.py      # severity scorer, batch analyzer, clustering, p50/p95
│   ├── api.py         # FastAPI app (dashboard, /health, /triage)
│   ├── cli.py         # CLI entrypoint (indic-triage)
│   └── static/dashboard.html
├── fixtures/           # strictly synthetic session JSONs (zero real data/PII)
├── tests/              # pytest suite (scripts, entities, triage, api, cli)
└── README.md
```
