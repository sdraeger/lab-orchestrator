from __future__ import annotations

import csv
import io
import json
import os
import shutil
import socket
import subprocess
import sys
from typing import Any

NVIDIA_SMI_TIMEOUT_S = 2.0


def main() -> None:
    payload = collect_local_snapshot()
    print(json.dumps(payload, sort_keys=True))


def collect_local_snapshot() -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore

    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    cpus_total = float(os.cpu_count() or 0)
    cpu_percent = 0.0
    memory_total_gb = 0.0
    memory_available_gb = 0.0

    if psutil is not None:
        cpu_percent = float(psutil.cpu_percent(interval=0.15))
        vm = psutil.virtual_memory()
        memory_total_gb = float(vm.total) / (1024**3)
        memory_available_gb = float(vm.available) / (1024**3)
    else:
        try:
            load1 = float(os.getloadavg()[0])
            cpu_percent = 100.0 * load1 / max(cpus_total, 1.0)
        except Exception:
            cpu_percent = 0.0

    gpu_util_avg = 0.0
    gpu_memory_free_gb = 0.0
    gpus_total = 0.0
    gpus_in_use = 0.0
    gpu_users: list[str] = []
    gpu_details: list[dict[str, Any]] = []

    if shutil.which("nvidia-smi"):
        gpu_rows = _query_nvidia_smi(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        proc_rows = _query_nvidia_smi(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )

        gpu_by_uuid: dict[str, dict[str, float]] = {}
        for row in gpu_rows:
            if len(row) < 5:
                continue
            uuid = row[1]
            gpu_by_uuid[uuid] = {
                "mem_total": _float_or_zero(row[2]),
                "mem_used": _float_or_zero(row[3]),
                "util": _float_or_zero(row[4]),
                "proc_count": 0.0,
            }

        user_set: set[str] = set()
        for row in proc_rows:
            if len(row) < 2:
                continue
            uuid = row[0]
            pid = _int_or_none(row[1])
            if uuid in gpu_by_uuid:
                gpu_by_uuid[uuid]["proc_count"] += 1.0
            if pid is not None and psutil is not None:
                try:
                    user_set.add(psutil.Process(pid).username())
                except Exception:
                    pass

        if gpu_by_uuid:
            gpus_total = float(len(gpu_by_uuid))
            gpu_util_avg = sum(row["util"] for row in gpu_by_uuid.values()) / float(
                len(gpu_by_uuid)
            )
            mem_free_mib = sum(
                max(0.0, row["mem_total"] - row["mem_used"])
                for row in gpu_by_uuid.values()
            )
            gpu_memory_free_gb = mem_free_mib / 1024.0
            gpus_in_use = float(
                sum(1 for row in gpu_by_uuid.values() if row["proc_count"] > 0.0)
            )
            gpu_users = sorted(user_set)
            gpu_details = sorted(
                [
                    {
                        "index": _int_or_none(row[0]),
                        "uuid": row[1],
                        "memory_total_mib": gpu_by_uuid[row[1]]["mem_total"],
                        "memory_used_mib": gpu_by_uuid[row[1]]["mem_used"],
                        "memory_free_mib": max(
                            0.0,
                            gpu_by_uuid[row[1]]["mem_total"]
                            - gpu_by_uuid[row[1]]["mem_used"],
                        ),
                        "util_percent": gpu_by_uuid[row[1]]["util"],
                        "proc_count": gpu_by_uuid[row[1]]["proc_count"],
                    }
                    for row in gpu_rows
                    if len(row) >= 2 and _int_or_none(row[0]) is not None
                ],
                key=lambda item: int(item["index"]),
            )

    return {
        "node_id": hostname,
        "ip": _primary_ip(),
        "hostname": hostname,
        "cpus_total": cpus_total,
        "gpus_total": gpus_total,
        "cpu_percent": max(0.0, min(cpu_percent, 100.0)),
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "gpu_util_avg": max(0.0, min(gpu_util_avg, 100.0)),
        "gpu_memory_free_gb": gpu_memory_free_gb,
        "gpus_in_use": gpus_in_use,
        "gpu_users": gpu_users,
        "extras": {"fqdn": fqdn, "gpu_details": gpu_details},
    }


def _primary_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


def _query_nvidia_smi(cmd: list[str]) -> list[list[str]]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    content = result.stdout.strip()
    if not content:
        return []
    reader = csv.reader(io.StringIO(content))
    return [[part.strip() for part in row] for row in reader]


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
