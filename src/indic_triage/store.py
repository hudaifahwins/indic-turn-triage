"""SQLite persistence for the Indic turn-triage engine."""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_sessions (
    session_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    issue_count INTEGER NOT NULL,
    turns_analyzed INTEGER NOT NULL,
    max_latency_ms INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS triage_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    code TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    turn_id INTEGER,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_issues_session ON triage_issues(session_id);
CREATE TABLE IF NOT EXISTS triage_latency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    p50 REAL,
    p95 REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SESSION_INSERT = """
INSERT OR REPLACE INTO triage_sessions(session_id, verdict, exit_code, outcome,
                                        issue_count, turns_analyzed, max_latency_ms,
                                        summary, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_ISSUE_INSERT = """
INSERT INTO triage_issues(session_id, code, category, severity, turn_id, detail)
VALUES (?, ?, ?, ?, ?, ?)
"""

_LATENCY_INSERT = """
INSERT INTO triage_latency(session_id, metric, p50, p95) VALUES (?, ?, ?, ?)
"""


class StoreError(Exception):
    """Raised for database-level problems."""


class Store:
    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = str(db_path)
        self._conn = self._connect()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextlib.contextmanager
    def _conn_ctx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init(self) -> None:
        with self._conn_ctx() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1')"
            )
            conn.commit()

    def save_report(self, report) -> None:
        with self._conn_ctx() as conn:
            conn.execute(
                _SESSION_INSERT,
                (
                    report.session_id, report.verdict, report.exit_code, report.outcome,
                    len(report.issues), report.turns_analyzed, report.max_latency_ms,
                    f"{len(report.issues)} issue(s)", "now",
                ),
            )
            for issue in report.issues:
                conn.execute(
                    _ISSUE_INSERT,
                    (report.session_id, issue.code, issue.category, issue.severity,
                     issue.turn_id, issue.detail),
                )
            conn.execute(
                _LATENCY_INSERT,
                (report.session_id, "first_token", report.latency_p50_first_token_ms,
                 report.latency_p95_first_token_ms),
            )
            conn.execute(
                _LATENCY_INSERT,
                (report.session_id, "response", report.latency_p50_response_ms,
                 report.latency_p95_response_ms),
            )
            conn.commit()

    def list_sessions(self) -> list[dict]:
        with self._conn_ctx() as conn:
            rows = conn.execute(
                "SELECT session_id, verdict, exit_code, outcome, issue_count, "
                "turns_analyzed, max_latency_ms, summary, created_at "
                "FROM triage_sessions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_report(self, session_id: str):
        with self._conn_ctx() as conn:
            row = conn.execute(
                "SELECT * FROM triage_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                raise StoreError(f"session not found in database: {session_id}")
            issue_rows = conn.execute(
                "SELECT code, category, severity, turn_id, detail FROM triage_issues "
                "WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return {"session": dict(row), "issues": [dict(r) for r in issue_rows]}

    def close(self) -> None:
        with contextlib.closing(self._conn) as conn:
            conn.close()
