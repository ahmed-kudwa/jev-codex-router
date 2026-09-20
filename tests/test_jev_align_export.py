from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "jev_align_export.py"


def test_redacts_credentials_and_paths() -> None:
    namespace = {}
    exec(SCRIPT.read_text(encoding="utf-8"), namespace)
    redact_text = namespace["redact_text"]
    value = (
        "Fix /Users/apple/project with token=sk-abcdefghijklmnopqrstuvwxyz1234 "
        "and apikey_22167a99daf45ce6423eacd78dec09b"
    )
    output = redact_text(value)
    assert "sk-" not in output
    assert "apikey_" not in output
    assert "/Users" not in output
    assert "[REDACTED]" in output
    assert "[PATH]" in output


def test_export_is_bounded_and_labelable(tmp_path: Path) -> None:
    source = tmp_path / "live.jsonl"
    source.write_text(
        json.dumps(
            {
                "at": "2026-09-20T00:00:00Z",
                "task": "Review /Users/apple/project with api_key=secret-value",
                "gate": "hive:routine",
                "step": "user_turn",
                "model": "gpt-5.6-luna",
                "effort": "medium",
                "conf": 0.87,
                "status": 200,
                "context_chars": 123,
                "context_items": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "dataset.csv"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--output", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "1 rows exported" in result.stdout
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["human_route_class"] == ""
    assert "secret-value" not in rows[0]["task_summary"]
    assert "/Users" not in rows[0]["task_summary"]
