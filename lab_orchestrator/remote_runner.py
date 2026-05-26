from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class RunnerConfig:
    job_id: str
    command: str
    argv: list[str]
    shell: bool
    workdir: str
    env: dict[str, str]
    log_path: str
    stderr_path: str
    state_path: str
    max_retries: int = 0
    retry_backoff_seconds: float = 5.0
    heartbeat_interval_seconds: float = 5.0

    @classmethod
    def load(cls, path: str | Path) -> "RunnerConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("runner config must be a JSON object")
        raw_argv = payload.get("argv") or []
        if not isinstance(raw_argv, list):
            raise ValueError("runner config field 'argv' must be a list when present")
        return cls(
            job_id=str(payload["job_id"]),
            command=str(payload["command"]),
            argv=[str(item) for item in raw_argv],
            shell=bool(payload.get("shell", True)),
            workdir=str(payload["workdir"]),
            env={str(k): str(v) for k, v in dict(payload.get("env") or {}).items()},
            log_path=str(payload["log_path"]),
            stderr_path=str(
                payload.get("stderr_path") or _default_stderr_path(str(payload["log_path"]))
            ),
            state_path=str(payload["state_path"]),
            max_retries=max(0, int(payload.get("max_retries", 0))),
            retry_backoff_seconds=max(
                0.0, float(payload.get("retry_backoff_seconds", 5.0))
            ),
            heartbeat_interval_seconds=max(
                0.1, float(payload.get("heartbeat_interval_seconds", 5.0))
            ),
        )


class RemoteRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.hostname = os.uname().nodename
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop_requested = False
        self._signal_count = 0
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._return_code: int | None = None
        self._last_heartbeat_at: str | None = None
        self._attempt = 0
        self._retries_used = 0
        self._last_error = ""
        self._stdout_handle: Any = None
        self._stderr_handle: Any = None

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

        stdout_path = Path(self.config.log_path)
        stderr_path = Path(self.config.stderr_path)
        state_path = Path(self.config.state_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_handle = open(stdout_path, "ab", buffering=0)
        self._stderr_handle = open(stderr_path, "ab", buffering=0)
        _chmod_best_effort(stdout_path, 0o600)
        _chmod_best_effort(stderr_path, 0o600)

        try:
            self._write_state(status="QUEUED", pid=None)
            attempt = 0
            while True:
                if self._stop_requested:
                    break

                self._attempt = attempt
                self._retries_used = max(0, attempt)
                if self._started_at is None:
                    self._started_at = utc_now_iso()

                self._proc = self._spawn_process()
                self._write_state(status="RUNNING", pid=self._proc.pid)
                rc = self._wait_with_heartbeat()
                self._return_code = int(rc)
                self._proc = None

                if rc == 0:
                    self._ended_at = utc_now_iso()
                    self._write_state(status="SUCCEEDED", pid=None)
                    return 0

                if self._stop_requested:
                    break

                self._last_error = f"attempt {attempt + 1} exited with return_code={rc}"
                if attempt >= self.config.max_retries:
                    self._ended_at = utc_now_iso()
                    self._write_state(status="FAILED", pid=None)
                    return int(rc)

                attempt += 1
                self._write_log(
                    (
                        f"[lab-orch] retrying command after failure "
                        f"(attempt={attempt}/{self.config.max_retries}, "
                        f"backoff_seconds={self.config.retry_backoff_seconds:.2f})\n"
                    )
                )
                t0 = time.time()
                while (time.time() - t0) < self.config.retry_backoff_seconds:
                    if self._stop_requested:
                        break
                    time.sleep(0.1)

            if self._ended_at is None:
                self._ended_at = utc_now_iso()
            if self._return_code is None:
                self._return_code = 143
            self._write_state(status="CANCELLED", pid=None)
            return int(self._return_code)
        finally:
            if self._stdout_handle is not None:
                try:
                    self._stdout_handle.flush()
                except Exception:
                    pass
                self._stdout_handle.close()
                self._stdout_handle = None
            if self._stderr_handle is not None:
                try:
                    self._stderr_handle.flush()
                except Exception:
                    pass
                self._stderr_handle.close()
                self._stderr_handle = None

    def _handle_stop(self, signum: int, _frame: Any) -> None:
        self._stop_requested = True
        self._signal_count += 1
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if self._signal_count <= 1:
                os.killpg(proc.pid, signum)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _spawn_process(self) -> subprocess.Popen[bytes]:
        merged_env = os.environ.copy()
        merged_env.update(self.config.env)
        if self.config.shell:
            args: str | list[str] = self.config.command
            shell = True
        else:
            if not self.config.argv:
                raise RuntimeError("runner config requested shell=false but argv is empty")
            args = list(self.config.argv)
            shell = False
        return subprocess.Popen(
            args,
            shell=shell,
            cwd=self.config.workdir,
            env=merged_env,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            preexec_fn=os.setsid,
        )

    def _wait_with_heartbeat(self) -> int:
        proc = self._proc
        if proc is None:
            return 1
        next_heartbeat = 0.0
        while True:
            rc = proc.poll()
            if rc is not None:
                return int(rc)
            now = time.time()
            if now >= next_heartbeat:
                self._write_state(status="RUNNING", pid=proc.pid)
                next_heartbeat = now + self.config.heartbeat_interval_seconds
            time.sleep(min(0.5, self.config.heartbeat_interval_seconds))

    def _write_log(self, text: str) -> None:
        if self._stderr_handle is None:
            return
        try:
            self._stderr_handle.write(text.encode("utf-8", errors="replace"))
        except Exception:
            pass

    def _write_state(self, status: str, pid: int | None) -> None:
        if status == "RUNNING":
            self._last_heartbeat_at = utc_now_iso()
        payload = {
            "job_id": self.config.job_id,
            "status": status,
            "pid": pid,
            "return_code": self._return_code,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "last_heartbeat_at": self._last_heartbeat_at,
            "attempt": self._attempt,
            "max_retries": self.config.max_retries,
            "retries_used": self._retries_used,
            "last_error": self._last_error,
            "host": self.hostname,
            "command": self.config.command,
            "workdir": self.config.workdir,
        }
        _write_json_atomic(Path(self.config.state_path), payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        _chmod_best_effort(path, 0o600)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _default_stderr_path(stdout_path: str) -> str:
    path = Path(stdout_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}.stderr{path.suffix}"))
    return str(path.with_name(path.name + ".stderr"))


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="lab-orchestrator remote runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    runner = RemoteRunner(RunnerConfig.load(args.config))
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
