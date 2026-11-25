"""indic-turn-triage: deterministic triage for multilingual Indian voice/chat agents.

The package analyzes conversational turn transcripts for:

* Code-switch, script drift, and romanized Indic heuristics (Hinglish / Tanglish).
* Entity, amount, and slot corruption (INR amounts, lakh/crore, phones, times, IDs).
* Latency spikes, terminal outcomes, batch clustering, and p50/p95 metrics.

Exposes a CLI entry point ``indic-triage`` and a FastAPI triage dashboard.
"""

__version__ = "0.1.0"
