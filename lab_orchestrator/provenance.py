from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import ray

from .models import JobRequest
from .utils import utc_now_iso


def collect_submission_metadata(
    request: JobRequest,
    scheduler_name: str,
    policy_name: str | None,
    submit_user: str,
) -> dict[str, Any]:
    request_hash = _request_hash(request)
    git_meta = _git_metadata(request.workdir)
    return {
        "submitted_at": utc_now_iso(),
        "submit_user": submit_user,
        "scheduler": scheduler_name,
        "policy": policy_name or "",
        "request_hash": request_hash,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ray_version": getattr(ray, "__version__", "unknown"),
        "workdir": str(Path(request.workdir).expanduser()),
        "distributed": bool(request.distributed),
        "requested_cpus": float(request.cpus),
        "requested_gpus": float(request.gpus),
        "retry": {
            "max_retries": int(request.max_retries),
            "backoff_seconds": float(request.retry_backoff_seconds),
        },
        "env_keys": sorted(request.env.keys()),
        "git": git_meta,
        "user_metadata": dict(request.metadata or {}),
    }


def _request_hash(request: JobRequest) -> str:
    payload = [
        request.name,
        request.command,
        str(request.cpus),
        str(request.gpus),
        str(request.workdir),
        str(bool(request.distributed)),
        str(int(request.max_retries)),
        str(float(request.retry_backoff_seconds)),
    ]
    env_items = [f"{k}={request.env[k]}" for k in sorted(request.env.keys())]
    payload.extend(env_items)
    encoded = "\n".join(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata(workdir: str) -> dict[str, Any]:
    cwd = Path(workdir).expanduser()
    if not cwd.exists():
        return {"available": False}
    git_dir = _run_git(["rev-parse", "--git-dir"], cwd=cwd)
    if git_dir is None:
        return {"available": False}

    commit = _run_git(["rev-parse", "HEAD"], cwd=cwd) or ""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd) or ""
    status = _run_git(["status", "--porcelain"], cwd=cwd) or ""
    remote = _run_git(["remote", "get-url", "origin"], cwd=cwd) or ""
    return {
        "available": True,
        "commit": commit.strip(),
        "branch": branch.strip(),
        "dirty": bool(status.strip()),
        "origin": remote.strip(),
    }


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
