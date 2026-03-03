from __future__ import annotations

import time
from pathlib import Path

import pytest

from lab_orchestrator.job_actor import ScriptJobActor


def _wait_terminal(actor: ScriptJobActor, timeout_s: float = 8.0) -> dict:
    deadline = time.time() + timeout_s
    terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
    while time.time() < deadline:
        state = actor.status()
        if str(state.get("status")) in terminal:
            return state
        time.sleep(0.05)
    raise TimeoutError("actor did not reach terminal state")


def test_actor_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "job.log"
    actor = ScriptJobActor(
        job_id="job-1",
        command="echo ignored",
        workdir=str(tmp_path),
        env={},
        log_path=str(log_path),
        max_retries=1,
        retry_backoff_seconds=0.1,
    )

    codes = iter([1, 0])

    class FakeProc:
        def __init__(self, rc: int):
            self.pid = 1234
            self._rc = rc

        def wait(self) -> int:
            return self._rc

    monkeypatch.setattr(actor, "_spawn_process", lambda: FakeProc(next(codes)))
    monkeypatch.setattr("lab_orchestrator.job_actor.time.sleep", lambda _s: None)

    actor.start()
    state = _wait_terminal(actor)
    assert state["status"] == "SUCCEEDED"
    assert int(state["retries_used"]) == 1
    assert "retrying command after failure" in actor.tail(200)


def test_actor_fails_without_retries(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    actor = ScriptJobActor(
        job_id="job-2",
        command="bash -lc 'exit 2'",
        workdir=str(tmp_path),
        env={},
        log_path=str(log_path),
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    actor.start()
    state = _wait_terminal(actor)
    assert state["status"] == "FAILED"
    assert int(state["retries_used"]) == 0
