"""Command line interface for indic-turn-triage.

Exit codes:
* ``0`` -- healthy (no HIGH / CRITICAL findings)
* ``1`` -- HIGH / CRITICAL failures found
* ``2`` -- invalid / malformed input or usage error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from .models import DEFAULT_MAX_LATENCY_MS, SEVERITY_RANK
from .triage import AnalysisResult, BatchResult, analyze_batch, analyze_payload

_MAIN = "indic-turn-triage"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indic-triage",
        description="Deterministic triage for multilingual Indian voice/chat agents.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", metavar="FILE", help="single session JSON file")
    source.add_argument("--input-dir", metavar="DIR", help="directory of session JSON files")
    parser.add_argument(
        "--max-latency-ms",
        type=int,
        default=DEFAULT_MAX_LATENCY_MS,
        help="latency budget in ms (default: 1500)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument("--output", metavar="FILE", help="write output to a file instead of stdout")
    return parser


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:g}ms"


def _render_issues(result: AnalysisResult, fmt: str) -> str:
    if not result.issues:
        return "  (no issues)"
    lines = []
    for issue in result.issues:
        location = f"turn #{issue.turn_index}" if issue.turn_index is not None else "session"
        if fmt == "markdown":
            lines.append(
                f"- **[{issue.severity}]** `{issue.category}` @ {location}: {issue.message}"
            )
        else:
            lines.append(f"  [{issue.severity}] {issue.category} @ {location}: {issue.message}")
    return "\n".join(lines)


def render_result(result: AnalysisResult, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result.to_dict(), indent=2)
    if fmt == "markdown":
        lines = [
            f"## Session `{result.session_id}`",
            "",
            f"- Verdict: **{result.verdict}** (exit {result.exit_code})",
            f"- First-token latency: p50={_fmt_ms(result.latency['first_token']['p50'])} "
            f"p95={_fmt_ms(result.latency['first_token']['p95'])}",
            f"- Response latency: p50={_fmt_ms(result.latency['response']['p50'])} "
            f"p95={_fmt_ms(result.latency['response']['p95'])}",
            "",
            "### Issues",
            "",
        ]
        lines.append(_render_issues(result, fmt))
        return "\n".join(lines)
    lines = [
        "=" * 72,
        f"{_MAIN} analysis",
        "=" * 72,
        f"Session    : {result.session_id}",
        f"Verdict   : {result.verdict} (exit {result.exit_code})",
        f"Latency   : first-token p50={_fmt_ms(result.latency['first_token']['p50'])} "
        f"p95={_fmt_ms(result.latency['first_token']['p95'])} | "
        f"response p50={_fmt_ms(result.latency['response']['p50'])} "
        f"p95={_fmt_ms(result.latency['response']['p95'])}",
        f"Issues    : {len(result.issues)}",
        "",
        _render_issues(result, fmt),
        "",
    ]
    return "\n".join(lines)


def render_batch(batch: BatchResult, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(batch.to_dict(), indent=2)
    if fmt == "markdown":
        lines = [
            f"## Batch analysis: `{batch.directory}`",
            "",
            f"- Sessions: {batch.total_sessions} valid, {batch.failed_sessions} failed, "
            f"{len(batch.skipped)} malformed (skipped)",
            f"- First-token latency: p50={_fmt_ms(batch.latency['first_token']['p50'])} "
            f"p95={_fmt_ms(batch.latency['first_token']['p95'])}",
            f"- Response latency: p50={_fmt_ms(batch.latency['response']['p50'])} "
            f"p95={_fmt_ms(batch.latency['response']['p95'])}",
            "",
            "### Failure clusters",
            "",
        ]
        if batch.clusters:
            lines.append("| Severity | Count | Root-cause signature | Sessions |")
            lines.append("| --- | --- | --- | --- |")
            for cluster in batch.clusters:
                lines.append(
                    f"| {cluster.severity} | {cluster.count} | {cluster.signature} | "
                    f"{', '.join(cluster.sessions)} |"
                )
        else:
            lines.append("_No failing sessions._")
        lines.append("")
        for result in batch.results:
            lines.append(render_result(result, "markdown"))
            lines.append("")
        return "\n".join(lines)
    lines = [
        "=" * 72,
        f"{_MAIN} batch analysis",
        "=" * 72,
        f"Directory  : {batch.directory}",
        f"Sessions   : {batch.total_sessions} valid, {batch.failed_sessions} failed, "
        f"{len(batch.skipped)} malformed (skipped)",
        f"Latency   : first-token p50={_fmt_ms(batch.latency['first_token']['p50'])} "
        f"p95={_fmt_ms(batch.latency['first_token']['p95'])} | "
        f"response p50={_fmt_ms(batch.latency['response']['p50'])} "
        f"p95={_fmt_ms(batch.latency['response']['p95'])}",
        "",
        "Failure clusters:",
    ]
    if batch.clusters:
        for cluster in batch.clusters:
            lines.append(
                f"  [{cluster.severity}] x{cluster.count} {cluster.signature} "
                f"-> {', '.join(cluster.sessions)}"
            )
    else:
        lines.append("  (none)")
    lines.append("")
    if batch.skipped:
        lines.append("Skipped malformed files:")
        for item in batch.skipped:
            lines.append(f"  - {item['file']}: {item['error']}")
        lines.append("")
    for result in batch.results:
        lines.append(render_result(result, "text"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.input_dir:
            directory = Path(args.input_dir)
            if not directory.is_dir():
                print(f"error: input directory not found: {directory}", file=sys.stderr)
                return 2
            batch = analyze_batch(directory, max_latency_ms=args.max_latency_ms)
            if not batch.results:
                print(
                    f"error: no valid session JSON files found in {directory}",
                    file=sys.stderr,
                )
                return 2
            exit_code = 1 if batch.failed_sessions > 0 else 0
            output = render_batch(batch, args.format)
        else:
            path = Path(args.input)
            if not path.is_file():
                print(f"error: input file not found: {path}", file=sys.stderr)
                return 2
            data = _load_json(path)
            result = analyze_payload(data, max_latency_ms=args.max_latency_ms)
            if result.exit_code == 2:
                print(f"error: malformed session input: {result.error}", file=sys.stderr)
                return 2
            exit_code = result.exit_code
            output = render_result(result, args.format)

        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return exit_code
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
