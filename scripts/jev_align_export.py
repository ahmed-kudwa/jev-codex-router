#!/usr/bin/env python3
"""Export a redacted, human-labelable routing dataset for jev-align.

This is intentionally offline and dependency-free. It reads the local router
trace, removes credentials and machine-specific paths, and writes bounded CSV
rows suitable for ``jeva optimize``. It never calls a model or changes live
routing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_LIVE_LOG = Path("~/.codex/codex-router/jev-router-live.jsonl").expanduser()
DEFAULT_OUTPUT = Path("~/.codex/codex-router/jev-align/routing-dataset.csv").expanduser()
MAX_TASK_CHARS = 240
MAX_ROWS = 5000

# Keep enough intent for a reviewer while preventing common credential leaks.
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(?:apikey|api_key|access_token|refresh_token)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)\b(?:x-api-key|authorization)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\b(?:ghp|gho|github_pat|glpat|xoxb|xoxp)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bAIza[0-9A-Za-z_-]{20,}\b"),
)
ABS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:Users|private|tmp|var|home|opt|etc|Volumes)(?:/[^\s`'\"]*)?")
HOME_PATH_RE = re.compile(r"(?:~/|/Users/[^\s`'\"]+)")
WHITESPACE_RE = re.compile(r"\s+")

FIELDNAMES = [
    "id",
    "task_summary",
    "route_class",
    "step_type",
    "model",
    "effort",
    "confidence",
    "status",
    "successful",
    "reason",
    "context_chars",
    "context_items",
    "digest_len",
    "cache_hit",
    "retried",
    "compaction_advised",
    "human_route_class",
    "human_effort",
]


def redact_text(value: str, *, limit: int = MAX_TASK_CHARS) -> str:
    """Redact credentials and host-specific paths from a bounded summary."""
    text = value if isinstance(value, str) else ""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = HOME_PATH_RE.sub("[PATH]", text)
    text = ABS_PATH_RE.sub("[PATH]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text[:limit]


def _scalar(row: dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    return default if value is None else value


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _stable_id(row: dict[str, Any], ordinal: int) -> str:
    material = json.dumps(
        {
            "task": redact_text(str(_scalar(row, "task"))),
            "at": _scalar(row, "at"),
            "model": _scalar(row, "model"),
            "status": _scalar(row, "status"),
            "ordinal": ordinal,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return "trace-" + hashlib.sha256(material).hexdigest()[:16]


def _route_class(row: dict[str, Any]) -> str:
    hive = row.get("hive")
    if isinstance(hive, dict):
        value = hive.get("route_class") or hive.get("routeClass")
        if isinstance(value, str) and value:
            return value
    gate = row.get("gate")
    if isinstance(gate, str) and gate.startswith("hive:"):
        return gate.removeprefix("hive:").split("(", 1)[0] or "other"
    return "other"


def row_from_trace(row: dict[str, Any], ordinal: int) -> dict[str, str]:
    status = _scalar(row, "status", 0)
    try:
        status_number = int(status)
    except (TypeError, ValueError):
        status_number = 0
    return {
        "id": _stable_id(row, ordinal),
        "task_summary": redact_text(str(_scalar(row, "task"))),
        "route_class": _route_class(row),
        "step_type": redact_text(str(_scalar(row, "step", "other")), limit=40),
        "model": redact_text(str(_scalar(row, "model", "unknown")), limit=100),
        "effort": redact_text(str(_scalar(row, "effort", "medium")), limit=20),
        "confidence": str(_scalar(row, "conf", "")),
        "status": str(status_number),
        "successful": _bool(status_number == 200),
        "reason": redact_text(str(_scalar(row, "gate", "")), limit=80),
        "context_chars": str(_scalar(row, "context_chars", 0)),
        "context_items": str(_scalar(row, "context_items", 0)),
        "digest_len": str(_scalar(row, "digest_len", 0)),
        "cache_hit": _bool(_scalar(row, "cacheHit", False)),
        "retried": _bool(_scalar(row, "retried", False)),
        "compaction_advised": _bool(_scalar(row, "compactionAdvised", False)),
        # Filled by a human in a copied/working dataset; never inferred from
        # transport success because HTTP 200 is not semantic correctness.
        "human_route_class": "",
        "human_effort": "",
    }


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeError):
                continue
            if isinstance(row, dict):
                yield row


def export_dataset(source: Path, destination: Path, *, limit: int = MAX_ROWS) -> tuple[int, int]:
    if limit < 1 or limit > MAX_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_ROWS}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    scanned = 0
    for ordinal, raw in enumerate(iter_rows(source), start=1):
        scanned += 1
        if len(rows) >= limit:
            break
        row = row_from_trace(raw, ordinal)
        if not row["task_summary"] and row["step_type"] == "other":
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        rows.append(row)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return scanned, len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_LIVE_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=MAX_ROWS)
    args = parser.parse_args(argv)
    source = args.source.expanduser()
    destination = args.output.expanduser()
    if not source.exists():
        parser.error(f"trace file does not exist: {source}")
    scanned, exported = export_dataset(source, destination, limit=args.limit)
    print(f"jev-align dataset: {exported} rows exported from {scanned} trace rows")
    print(f"output: {destination}")
    print("next: jeva optimize <output> --question 'Choose the correct route class' --column task_summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
