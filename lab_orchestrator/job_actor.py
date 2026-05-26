from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Any


class ScriptJobActor:
    def __init__(
        self,
        job_id: str,
        command: str,
        workdir: str,
        env: dict[str, str],
        log_path: str,
        command_argv: list[str] | None = None,
        use_shell: bool = True,
        max_retries: int = 0,
        retry_backoff_seconds: float = 5.0,
    ):
        self.job_id = job_id
        self.command = command
        self.command_argv = list(command_argv or [])
        self.use_shell = bool(use_shell)
        self.workdir = workdir
        self.env = env
        self.log_path = log_path
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

        self._proc: subprocess.Popen[bytes] | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._stop_requested = False
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._return_code: int | None = None
        self._log_handle: Any = None
        self._attempt = 0
        self._retries_used = 0
        self._last_error = ""
        self._lock = threading.Lock()

    def start(self) -> dict:
        already_started = False
        with self._lock:
            if self._supervisor_thread is not None:
                already_started = True
            else:
                self._stop_requested = False
                self._started_at = time.time()
        if already_started:
            return self.status()

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_handle = open(self.log_path, "ab", buffering=0)
        self._supervisor_thread = threading.Thread(
            target=self._run_supervisor,
            name=f"lab-orch-supervisor-{self.job_id}",
            daemon=True,
        )
        self._supervisor_thread.start()
        return self.status()

    def _run_supervisor(self) -> None:
        attempt = 0
        while True:
            with self._lock:
                if self._stop_requested:
                    break
                self._attempt = attempt
                self._retries_used = max(0, attempt)
                self._proc = self._spawn_process()

            rc = -1
            proc = self._proc
            if proc is not None:
                rc = proc.wait()

            with self._lock:
                self._return_code = rc
                self._proc = None

            if rc == 0:
                break

            with self._lock:
                if self._stop_requested:
                    break
                self._last_error = f"attempt {attempt + 1} exited with return_code={rc}"

            if attempt >= self.max_retries:
                break

            attempt += 1
            self._write_log(
                (
                    f"[lab-orch] retrying command after failure "
                    f"(attempt={attempt}/{self.max_retries}, "
                    f"backoff_seconds={self.retry_backoff_seconds:.2f})\n"
                )
            )
            t0 = time.time()
            while (time.time() - t0) < self.retry_backoff_seconds:
                with self._lock:
                    if self._stop_requested:
                        break
                time.sleep(0.1)
            with self._lock:
                if self._stop_requested:
                    break

        with self._lock:
            if self._ended_at is None:
                self._ended_at = time.time()
            if self._log_handle is not None:
                try:
                    self._log_handle.flush()
                except Exception:
                    pass
                self._log_handle.close()
                self._log_handle = None

    def _spawn_process(self) -> subprocess.Popen[bytes]:
        merged_env = os.environ.copy()
        merged_env.update(self.env)
        if self.use_shell:
            args: str | list[str] = self.command
            shell = True
        else:
            if not self.command_argv:
                raise RuntimeError("ScriptJobActor requested shell=false but command_argv is empty")
            args = list(self.command_argv)
            shell = False
        return subprocess.Popen(
            args,
            shell=shell,
            cwd=self.workdir,
            env=merged_env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def _write_log(self, text: str) -> None:
        handle = self._log_handle
        if handle is None:
            return
        try:
            handle.write(text.encode("utf-8", errors="replace"))
        except Exception:
            pass

    def status(self) -> dict:
        with self._lock:
            if self._supervisor_thread is None:
                return {
                    "status": "QUEUED",
                    "pid": None,
                    "return_code": None,
                    "started_at": self._started_at,
                    "ended_at": self._ended_at,
                    "attempt": self._attempt,
                    "max_retries": self.max_retries,
                    "retries_used": self._retries_used,
                    "last_error": self._last_error,
                }

            proc = self._proc
            return_code = self._return_code
            started_at = self._started_at
            ended_at = self._ended_at
            attempt = self._attempt
            retries_used = self._retries_used
            last_error = self._last_error
            stop_requested = self._stop_requested

        if proc is None and ended_at is None:
            return {
                "status": "RUNNING",
                "pid": None,
                "return_code": return_code,
                "started_at": started_at,
                "ended_at": None,
                "attempt": attempt,
                "max_retries": self.max_retries,
                "retries_used": retries_used,
                "last_error": last_error,
            }

        if proc is not None:
            return {
                "status": "RUNNING",
                "pid": proc.pid,
                "return_code": None,
                "started_at": started_at,
                "ended_at": None,
                "attempt": attempt,
                "max_retries": self.max_retries,
                "retries_used": retries_used,
                "last_error": last_error,
            }

        final_status = "SUCCEEDED" if int(return_code or 0) == 0 else "FAILED"
        if stop_requested and final_status != "SUCCEEDED":
            final_status = "CANCELLED"
        return {
            "status": final_status,
            "pid": None,
            "return_code": return_code,
            "started_at": started_at,
            "ended_at": ended_at,
            "attempt": attempt,
            "max_retries": self.max_retries,
            "retries_used": retries_used,
            "last_error": last_error,
        }

    def stop(self, grace_seconds: int = 20) -> dict:
        with self._lock:
            self._stop_requested = True
            proc = self._proc

        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                t0 = time.time()
                while time.time() - t0 < grace_seconds:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        thread = self._supervisor_thread
        if thread is not None:
            thread.join(timeout=max(1.0, float(grace_seconds) + 1.0))

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
