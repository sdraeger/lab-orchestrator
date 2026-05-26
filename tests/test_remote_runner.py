from __future__ import annotations

import json
import sys
from pathlib import Path

from lab_orchestrator.remote_runner import main


def test_remote_runner_success(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    stderr_path = tmp_path / "job.stderr.log"
    state_path = tmp_path / "job.json"
    cfg_path = tmp_path / "runner.json"
    cfg_path.write_text(
        json.dumps(
            {
                "job_id": "j1",
                "command": "python -c \"import sys; print('hello'); print('oops', file=sys.stderr)\"",
                "workdir": str(tmp_path),
                "env": {},
                "log_path": str(log_path),
                "stderr_path": str(stderr_path),
                "state_path": str(state_path),
                "max_retries": 0,
                "retry_backoff_seconds": 0.0,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "SUCCEEDED"
    assert "hello" in log_path.read_text(encoding="utf-8")
    assert "oops" in stderr_path.read_text(encoding="utf-8")


def test_remote_runner_retry(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    log_path = tmp_path / "job.log"
    stderr_path = tmp_path / "job.stderr.log"
    state_path = tmp_path / "job.json"
    cfg_path = tmp_path / "runner.json"
    command = (
        "/bin/sh -lc '"
        "echo attempt; "
        "if [ -f marker.txt ]; then exit 0; fi; "
        "touch marker.txt; "
        "exit 1'"
    )
    cfg_path.write_text(
        json.dumps(
            {
                "job_id": "j2",
                "command": command,
                "workdir": str(tmp_path),
                "env": {},
                "log_path": str(log_path),
                "stderr_path": str(stderr_path),
                "state_path": str(state_path),
                "max_retries": 1,
                "retry_backoff_seconds": 0.0,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--config", str(cfg_path)])
    assert rc == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "SUCCEEDED"
    assert state["retries_used"] == 1
    assert marker.exists()
    assert "retrying command after failure" in stderr_path.read_text(encoding="utf-8")


def test_remote_runner_argv_mode_and_heartbeat(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    stderr_path = tmp_path / "job.stderr.log"
    state_path = tmp_path / "job.json"
    cfg_path = tmp_path / "runner.json"
    cfg_path.write_text(
        json.dumps(
            {
                "job_id": "j3",
                "command": "unused shell string",
                "argv": [
                    sys.executable,
                    "-c",
                    "import os, time; print(os.environ['LAB_ARGV_TEST']); time.sleep(0.05)",
                ],
                "shell": False,
                "workdir": str(tmp_path),
                "env": {"LAB_ARGV_TEST": "argv-ok"},
                "log_path": str(log_path),
                "stderr_path": str(stderr_path),
                "state_path": str(state_path),
                "heartbeat_interval_seconds": 0.01,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--config", str(cfg_path)])

    assert rc == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "SUCCEEDED"
    assert state["last_heartbeat_at"]
    assert "argv-ok" in log_path.read_text(encoding="utf-8")
