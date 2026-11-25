"""End-to-end tests for the CLI (exit codes and output formats)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "indic_triage.cli", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )


def test_single_healthy_exits_0():
    result = run("--input", str(FIXTURES / "healthy_hinglish_session.json"))
    assert result.returncode == 0
    assert "HEALTHY" in result.stdout
    assert "healthy_hinglish_session" in result.stdout


def test_single_failing_exits_1():
    result = run("--input", str(FIXTURES / "failing_session_1.json"))
    assert result.returncode == 1
    assert "FAILED" in result.stdout
    assert "CRITICAL" in result.stdout


def test_single_malformed_exits_2():
    result = run("--input", str(FIXTURES / "malformed_session.json"))
    assert result.returncode == 2
    assert "malformed" in result.stderr.lower()


def test_missing_file_exits_2():
    result = run("--input", str(ROOT / "nope.json"))
    assert result.returncode == 2


def test_batch_directory_exits_1():
    result = run("--input-dir", str(FIXTURES))
    assert result.returncode == 1
    out = result.stdout.lower()
    assert "batch analysis" in out
    assert "failure clusters" in out
    assert "malformed" in out  # skipped-file warning present


def test_json_format_output():
    result = run(
        "--input",
        str(FIXTURES / "healthy_hinglish_session.json"),
        "--format",
        "json",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["session_id"] == "healthy_hinglish_session"
    assert data["exit_code"] == 0
    assert data["issues"] == []


def test_markdown_format_output():
    result = run(
        "--input",
        str(FIXTURES / "failing_session_1.json"),
        "--format",
        "markdown",
    )
    assert result.returncode == 1
    assert "## Session" in result.stdout
    assert "**[CRITICAL]**" in result.stdout


def test_output_file(tmp_path):
    out = tmp_path / "report.txt"
    result = run(
        "--input",
        str(FIXTURES / "healthy_hinglish_session.json"),
        "--output",
        str(out),
    )
    assert result.returncode == 0
    assert out.exists()
    assert "HEALTHY" in out.read_text(encoding="utf-8")


def test_max_latency_flag_affects_exit():
    healthy = str(FIXTURES / "healthy_hinglish_session.json")
    normal = run("--input", healthy)
    assert normal.returncode == 0
    strict = run("--input", healthy, "--max-latency-ms", "100")
    assert strict.returncode == 1
    assert "latency_spike" in strict.stdout
