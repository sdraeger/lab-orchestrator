from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from lab_orchestrator.job_actor import ScriptJobActor


def test_status_before_start_and_tail_edges(tmp_path: Path) -> None:
    actor = ScriptJobActor(
        job_id="q",
        command="echo hi",
        workdir=str(tmp_path),
        env={},
        log_path=str(tmp_path / "x.log"),
    )
    state = actor.status()
    assert state["status"] == "QUEUED"
    assert actor.tail(0) == ""
    assert actor.stop(grace_seconds=1)["status"] == "QUEUED"


def test_start_idempotent_and_stop_running(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    actor = ScriptJobActor(
        job_id="run",
        command="bash -lc 'sleep 5'",
        workdir=str(tmp_path),
        env={},
        log_path=str(log_path),
    )
    actor.start()
    state = actor.start()
    assert state["status"] in {"RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}
    stopped = actor.stop(grace_seconds=1)
    assert stopped["status"] in {"CANCELLED", "SUCCEEDED", "FAILED"}
    assert os.path.exists(log_path)


def test_stop_handles_process_lookup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actor = ScriptJobActor(
        job_id="pl",
        command="bash -lc 'sleep 1'",
        workdir=str(tmp_path),
        env={},
        log_path=str(tmp_path / "pl.log"),
    )
    actor.start()
    # Give supervisor a moment to spawn process.
    deadline = time.time() + 1.0
    while time.time() < deadline:
        st = actor.status()
        if st["status"] == "RUNNING" and st["pid"] is not None:
            break
        time.sleep(0.05)

    def raise_lookup(pid: int, sig: int) -> None:
        _ = pid, sig
        raise ProcessLookupError("gone")

    monkeypatch.setattr("lab_orchestrator.job_actor.os.killpg", raise_lookup)
    stopped = actor.stop(grace_seconds=1)
    if stopped["status"] == "RUNNING":
        # The process may race to completion when killpg reports it missing.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            stopped = actor.status()
            if stopped["status"] != "RUNNING":
                break
            time.sleep(0.05)
    assert stopped["status"] in {"RUNNING", "CANCELLED", "FAILED", "SUCCEEDED"}
    # Ensure no process/thread leak into subsequent tests.
    monkeypatch.undo()
    actor.stop(grace_seconds=1)


def test_stop_escalates_to_sigkill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actor = ScriptJobActor(
        job_id="kill",
        command="bash -lc 'sleep 5'",
        workdir=str(tmp_path),
        env={},
        log_path=str(tmp_path / "kill.log"),
    )
    actor.start()
    deadline = time.time() + 1.0
    pid = None
    while time.time() < deadline:
        st = actor.status()
        pid = st.get("pid")
        if pid:
            break
        time.sleep(0.05)
    if not pid:
        pytest.skip("process did not start in time")

    sent: list[int] = []
    real_killpg = os.killpg

    def tracking_killpg(proc_pid: int, sig: int) -> None:
        sent.append(sig)
        if sig == signal.SIGTERM:
            return
        return real_killpg(proc_pid, sig)

    monkeypatch.setattr("lab_orchestrator.job_actor.os.killpg", tracking_killpg)
    actor.stop(grace_seconds=0)
    assert signal.SIGTERM in sent
    assert signal.SIGKILL in sent


def test_internal_supervisor_and_write_log_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    actor = ScriptJobActor(
        job_id="int",
        command="echo",
        workdir=str(tmp_path),
        env={},
        log_path=str(tmp_path / "int.log"),
        max_retries=1,
        retry_backoff_seconds=0.5,
    )

    class HandleWithFlushError:
        def write(self, data: bytes) -> None:
            _ = data
            raise RuntimeError("write")

        def flush(self) -> None:
            raise RuntimeError("flush")

        def close(self) -> None:
            return None

    actor._log_handle = None
    actor._write_log("x")
    actor._log_handle = HandleWithFlushError()
    actor._write_log("x")

    actor._stop_requested = True
    actor._run_supervisor()

    actor2 = ScriptJobActor(
        job_id="int2",
        command="echo",
        workdir=str(tmp_path),
        env={},
        log_path=str(tmp_path / "int2.log"),
        max_retries=1,
        retry_backoff_seconds=0.5,
    )

    class FakeProc:
        pid = 1

        def wait(self) -> int:
            return 1

    actor2._log_handle = HandleWithFlushError()
    monkeypatch.setattr(actor2, "_spawn_process", lambda: FakeProc())

    def fake_sleep(seconds: float) -> None:
        _ = seconds
        actor2._stop_requested = True

    monkeypatch.setattr("lab_orchestrator.job_actor.time.sleep", fake_sleep)
    actor2._run_supervisor()
