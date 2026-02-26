from __future__ import annotations

import os
import signal
import subprocess
import time
from collections import deque


class ScriptJobActor:
    def __init__(
        self,
        job_id: str,
        command: str,
        workdir: str,
        env: dict[str, str],
        log_path: str,
    ):
        self.job_id = job_id
        self.command = command
        self.workdir = workdir
        self.env = env
        self.log_path = log_path

        self._proc: subprocess.Popen[bytes] | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._return_code: int | None = None
        self._log_handle = None

    def start(self) -> dict:
        if self._proc is not None:
            return self.status()

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_handle = open(self.log_path, "ab", buffering=0)

        merged_env = os.environ.copy()
        merged_env.update(self.env)

        self._proc = subprocess.Popen(
            self.command,
            shell=True,
            cwd=self.workdir,
            env=merged_env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._started_at = time.time()
        return self.status()

    def status(self) -> dict:
        if self._proc is None:
            return {
                "status": "QUEUED",
                "pid": None,
                "return_code": None,
                "started_at": self._started_at,
                "ended_at": self._ended_at,
            }

        rc = self._proc.poll()
        if rc is None:
            return {
                "status": "RUNNING",
                "pid": self._proc.pid,
                "return_code": None,
                "started_at": self._started_at,
                "ended_at": None,
            }

        self._return_code = rc
        if self._ended_at is None:
            self._ended_at = time.time()
            if self._log_handle is not None:
                self._log_handle.flush()
                self._log_handle.close()
                self._log_handle = None

        return {
            "status": "SUCCEEDED" if rc == 0 else "FAILED",
            "pid": self._proc.pid,
            "return_code": rc,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
        }

    def stop(self, grace_seconds: int = 20) -> dict:
        if self._proc is None:
            return self.status()

        if self._proc.poll() is not None:
            return self.status()

        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
            t0 = time.time()
            while time.time() - t0 < grace_seconds:
                if self._proc.poll() is not None:
                    break
                time.sleep(0.2)
            if self._proc.poll() is None:
                os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        return self.status()

    def tail(self, n_lines: int = 100) -> str:
        if n_lines < 1:
            n_lines = 1
        if not os.path.exists(self.log_path):
            return ""
        lines: deque[str] = deque(maxlen=n_lines)
        with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines.append(line)
        return "".join(lines)
