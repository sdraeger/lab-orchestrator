from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_smoke_script_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "smoke.json"
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "test.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--label",
            "pytest-smoke",
            "--output-json",
            str(output_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["label"] == "pytest-smoke"
    assert payload["ok"] is True
    assert output_path.exists()


def test_smoke_script_expect_host_failure() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "test.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--expect-host",
            "definitely-not-this-host",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert "expected hostname" in result.stderr
