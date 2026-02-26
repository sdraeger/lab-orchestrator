from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import yaml


def _ssh_run(host: str, cmd: str, ssh_user: str | None, dry_run: bool) -> None:
    target = f"{ssh_user}@{host}" if ssh_user else host
    ssh_cmd = ["ssh", target, cmd]
    if dry_run:
        print("DRY-RUN:", " ".join(shlex.quote(x) for x in ssh_cmd))
        return
    subprocess.run(ssh_cmd, check=True)


def _ray_exec_prefix(ray_bin: str | None) -> str:
    # Resolve ray binary robustly on remote host even with minimal non-interactive PATH.
    if ray_bin:
        preferred = shlex.quote(ray_bin)
        return (
            f"if [ -x {preferred} ]; then RAY_BIN={preferred}; "
            'elif command -v ray >/dev/null 2>&1; then RAY_BIN="$(command -v ray)"; '
            'elif [ -x "$HOME/miniconda3/bin/ray" ]; then RAY_BIN="$HOME/miniconda3/bin/ray"; '
            'else echo "ray binary not found" >&2; exit 127; fi; '
        )
    return (
        'if command -v ray >/dev/null 2>&1; then RAY_BIN="$(command -v ray)"; '
        'elif [ -x "$HOME/miniconda3/bin/ray" ]; then RAY_BIN="$HOME/miniconda3/bin/ray"; '
        'else echo "ray binary not found" >&2; exit 127; fi; '
    )


def bootstrap_cluster(config_path: str, dry_run: bool = False) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Invalid bootstrap config: expected a YAML mapping")

    head = str(config.get("head", "")).strip()
    head_address = str(config.get("head_address", head)).strip()
    workers = config.get("workers", [])
    ssh_user = config.get("ssh_user")
    ray_port = int(config.get("ray_port", 6379))
    dashboard_port = int(config.get("dashboard_port", 8265))
    ray_bin = config.get("ray_bin")

    if not head:
        raise ValueError("bootstrap config requires a non-empty 'head' field")
    if not head_address:
        raise ValueError(
            "bootstrap config requires a non-empty 'head_address' (or valid 'head')"
        )
    if not isinstance(workers, list):
        raise ValueError("bootstrap config field 'workers' must be a YAML list")

    head_start = (
        _ray_exec_prefix(ray_bin)
        + '"$RAY_BIN" stop --force >/dev/null 2>&1; '
        + f'"$RAY_BIN" start --head --port={ray_port} --dashboard-host=0.0.0.0 --dashboard-port={dashboard_port}'
    )
    _ssh_run(head, head_start, ssh_user=ssh_user, dry_run=dry_run)

    for worker in workers:
        worker_host = str(worker).strip()
        if not worker_host:
            continue
        worker_start = (
            _ray_exec_prefix(ray_bin)
            + '"$RAY_BIN" stop --force >/dev/null 2>&1; '
            + f"\"$RAY_BIN\" start --address='{head_address}:{ray_port}'"
        )
        _ssh_run(worker_host, worker_start, ssh_user=ssh_user, dry_run=dry_run)


def stop_cluster(config_path: str, dry_run: bool = False) -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Invalid bootstrap config: expected a YAML mapping")

    head = str(config.get("head", "")).strip()
    workers = config.get("workers", [])
    ssh_user = config.get("ssh_user")
    ray_bin = config.get("ray_bin")

    hosts = [h for h in [head, *workers] if str(h).strip()]
    for host in hosts:
        _ssh_run(
            str(host),
            _ray_exec_prefix(ray_bin) + '"$RAY_BIN" stop --force',
            ssh_user=ssh_user,
            dry_run=dry_run,
        )
