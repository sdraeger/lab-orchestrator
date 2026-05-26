from __future__ import annotations

import json
import shlex
import socket
import subprocess
from pathlib import Path
import time

from .registry import MachineSpec

SYSTEMD_SHOW_FIELDS = [
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainPID",
    "ExecMainStatus",
    "MainPID",
    "NRestarts",
]

DEFAULT_UNIT_PROPERTIES = {
    "Type": "exec",
    "KillMode": "control-group",
    "TimeoutStopSec": "25s",
    "TasksMax": "4096",
}


def remote_probe_script_path() -> Path:
    return Path(__file__).with_name("remote_probe.py").resolve()


def remote_runner_script_path() -> Path:
    return Path(__file__).with_name("remote_runner.py").resolve()


class SSHSystemdTransport:
    def __init__(self, ssh_connect_timeout_s: float = 8.0):
        self.ssh_connect_timeout_s = max(1.0, float(ssh_connect_timeout_s))

    def probe_machine(self, machine: MachineSpec, timeout_s: float = 12.0) -> dict:
        cmd = shlex.join(
            [machine.python_bin, str(remote_probe_script_path())]
        )
        result = self.run_command(machine, cmd, timeout_s=timeout_s)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or f"probe failed with rc={result.returncode}")
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except Exception as exc:
            raise RuntimeError("probe returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("probe returned a non-object payload")
        return payload

    def start_unit(
        self,
        machine: MachineSpec,
        unit_name: str,
        config_path: str,
        description: str,
        timeout_s: float = 15.0,
        properties: dict[str, str] | None = None,
    ) -> str:
        unit_properties = dict(DEFAULT_UNIT_PROPERTIES)
        unit_properties.update({str(k): str(v) for k, v in (properties or {}).items()})
        argv = [
            "systemd-run",
            "--user",
            "--unit",
            unit_name,
            "--description",
            description,
            "--collect",
            "--setenv",
            "PYTHONUNBUFFERED=1",
            machine.python_bin,
            str(remote_runner_script_path()),
            "--config",
            config_path,
        ]
        property_args: list[str] = []
        for key, value in unit_properties.items():
            if value:
                property_args.extend(["--property", f"{key}={value}"])
        argv[7:7] = property_args
        result = self._run_script(
            machine,
            _with_systemd_env(shlex.join(argv)),
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"systemd-run failed with rc={result.returncode}"
            raise RuntimeError(detail)
        return unit_name

    def show_unit(
        self, machine: MachineSpec, unit_name: str, timeout_s: float = 5.0
    ) -> dict[str, str] | None:
        argv = [
            "systemctl",
            "--user",
            "show",
            unit_name,
            *[f"--property={field}" for field in SYSTEMD_SHOW_FIELDS],
        ]
        result = self._run_script(
            machine,
            _with_systemd_env(shlex.join(argv)),
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip().lower()
            stdout = (result.stdout or "").strip().lower()
            merged = f"{stderr}\n{stdout}"
            if "could not be found" in merged or "not loaded" in merged:
                return None
            raise RuntimeError(
                (result.stderr or result.stdout or "systemctl show failed").strip()
            )
        payload: dict[str, str] = {}
        for line in (result.stdout or "").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            payload[key.strip()] = value.strip()
        return payload

    def stop_unit(
        self, machine: MachineSpec, unit_name: str, timeout_s: float = 10.0
    ) -> None:
        argv = ["systemctl", "--user", "stop", unit_name]
        result = self._run_script(
            machine,
            _with_systemd_env(shlex.join(argv)),
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip().lower()
            stdout = (result.stdout or "").strip().lower()
            merged = f"{stderr}\n{stdout}"
            if "could not be found" in merged or "not loaded" in merged:
                return
            raise RuntimeError(
                (result.stderr or result.stdout or "systemctl stop failed").strip()
            )

    def run_command(
        self, machine: MachineSpec, command: str, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        return self._run_script(machine=machine, script=command, timeout_s=timeout_s)

    def doctor_machine(
        self, machine: MachineSpec, probe_timeout_s: float = 12.0
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "host": machine.host,
            "target": machine.connect_host,
            "mode": "local" if _is_local_machine(machine) else "ssh",
        }

        ssh_started = time.monotonic()
        ssh_res = self.run_command(machine, "hostname", timeout_s=6.0)
        ssh_elapsed = time.monotonic() - ssh_started
        result["ssh_ok"] = ssh_res.returncode == 0
        result["ssh_seconds"] = round(ssh_elapsed, 2)
        result["ssh_detail"] = (
            (ssh_res.stdout or ssh_res.stderr or "").strip()
            if ssh_res.returncode == 0
            else (ssh_res.stderr or ssh_res.stdout or "").strip()
        )
        if ssh_res.returncode != 0:
            return result

        py_started = time.monotonic()
        py_cmd = shlex.join(
            [machine.python_bin, "-c", "import sys; print(sys.executable)"]
        )
        py_res = self.run_command(machine, py_cmd, timeout_s=6.0)
        py_elapsed = time.monotonic() - py_started
        result["python_ok"] = py_res.returncode == 0
        result["python_seconds"] = round(py_elapsed, 2)
        result["python_detail"] = (
            (py_res.stdout or py_res.stderr or "").strip()
            if py_res.returncode == 0
            else (py_res.stderr or py_res.stdout or "").strip()
        )
        if py_res.returncode != 0:
            return result

        systemd_started = time.monotonic()
        sys_res = self.run_command(
            machine,
            _with_systemd_env("systemctl --user is-system-running || true"),
            timeout_s=8.0,
        )
        systemd_elapsed = time.monotonic() - systemd_started
        systemd_text = (sys_res.stdout or sys_res.stderr or "").strip()
        result["systemd_ok"] = bool(systemd_text) and "failed to connect" not in systemd_text.lower()
        result["systemd_seconds"] = round(systemd_elapsed, 2)
        result["systemd_detail"] = systemd_text or f"rc={sys_res.returncode}"
        if not bool(result["systemd_ok"]):
            return result

        probe_started = time.monotonic()
        try:
            payload = self.probe_machine(machine, timeout_s=probe_timeout_s)
            probe_elapsed = time.monotonic() - probe_started
            result["probe_ok"] = True
            result["probe_seconds"] = round(probe_elapsed, 2)
            result["probe_detail"] = (
                f"host={payload.get('hostname')} ip={payload.get('ip')} "
                f"cpus={float(payload.get('cpus_total', 0.0)):.0f} "
                f"gpus={float(payload.get('gpus_total', 0.0)):.0f}"
            )
        except Exception as exc:
            probe_elapsed = time.monotonic() - probe_started
            result["probe_ok"] = False
            result["probe_seconds"] = round(probe_elapsed, 2)
            result["probe_detail"] = str(exc)
        return result

    def _run_script(
        self, machine: MachineSpec, script: str, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        if _is_local_machine(machine):
            return subprocess.run(
                ["bash", "-lc", script],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1.0, float(timeout_s)),
            )

        target = machine.connect_host
        if machine.ssh_user:
            target = f"{machine.ssh_user}@{target}"
        ssh_cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(max(1.0, self.ssh_connect_timeout_s))}",
            target,
            script,
        ]
        return subprocess.run(
            ssh_cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(timeout_s)),
        )


def _with_systemd_env(command: str) -> str:
    return (
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)"; '
        'export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"; '
        + command
    )


def _is_local_machine(machine: MachineSpec) -> bool:
    if machine.local:
        return True
    host = machine.connect_host.strip().lower()
    aliases = {
        "localhost",
        "127.0.0.1",
        socket.gethostname().strip().lower(),
        socket.getfqdn().strip().lower(),
    }
    return host in aliases
