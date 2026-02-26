from __future__ import annotations

import csv
import io
import shutil
import socket
import subprocess
from typing import Any

import ray

from .models import NodeSnapshot


@ray.remote
def _probe_local(
    node_id: str, node_ip: str, cpus_total: float, gpus_total: float
) -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore

    hostname = socket.gethostname()

    if psutil is not None:
        cpu_percent = float(psutil.cpu_percent(interval=0.15))
        vm = psutil.virtual_memory()
        memory_total_gb = float(vm.total) / (1024**3)
        memory_available_gb = float(vm.available) / (1024**3)
    else:
        cpu_percent = 0.0
        memory_total_gb = 0.0
        memory_available_gb = 0.0

    gpu_util_avg = 0.0
    gpu_memory_free_gb = 0.0
    gpus_in_use = 0.0
    gpu_users: list[str] = []

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
            mem_total = float_or_zero(row[2])
            mem_used = float_or_zero(row[3])
            util = float_or_zero(row[4])
            gpu_by_uuid[uuid] = {
                "mem_total": mem_total,
                "mem_used": mem_used,
                "util": util,
                "proc_count": 0.0,
            }

        user_set: set[str] = set()
        for row in proc_rows:
            if len(row) < 2:
                continue
            uuid = row[0]
            pid = int_or_none(row[1])
            if uuid in gpu_by_uuid:
                gpu_by_uuid[uuid]["proc_count"] += 1.0
            if pid is not None and psutil is not None:
                try:
                    user_set.add(psutil.Process(pid).username())
                except Exception:
                    pass

        if gpu_by_uuid:
            gpus_total = float(len(gpu_by_uuid))
            util_sum = sum(v["util"] for v in gpu_by_uuid.values())
            gpu_util_avg = util_sum / float(len(gpu_by_uuid))
            mem_free_mib = sum(
                max(0.0, v["mem_total"] - v["mem_used"]) for v in gpu_by_uuid.values()
            )
            gpu_memory_free_gb = mem_free_mib / 1024.0
            gpus_in_use = float(
                sum(1 for v in gpu_by_uuid.values() if v["proc_count"] > 0.0)
            )
            gpu_users = sorted(user_set)

    return {
        "node_id": node_id,
        "ip": node_ip,
        "hostname": hostname,
        "cpus_total": float(cpus_total),
        "gpus_total": float(gpus_total),
        "cpu_percent": cpu_percent,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "gpu_util_avg": gpu_util_avg,
        "gpu_memory_free_gb": gpu_memory_free_gb,
        "gpus_in_use": gpus_in_use,
        "gpu_users": gpu_users,
    }


def _query_nvidia_smi(cmd: list[str]) -> list[list[str]]:
    try:
        res = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if res.returncode != 0:
        return []
    content = res.stdout.strip()
    if not content:
        return []
    reader = csv.reader(io.StringIO(content))
    rows: list[list[str]] = []
    for row in reader:
        rows.append([v.strip() for v in row])
    return rows


def float_or_zero(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def collect_cluster_snapshots() -> list[NodeSnapshot]:
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    probe_refs = []
    for node in nodes:
        resources = node.get("Resources", {})
        node_id = str(node.get("NodeID") or "")
        node_ip = str(node.get("NodeManagerAddress") or "")
        cpus_total = float(resources.get("CPU", 0.0))
        gpus_total = float(resources.get("GPU", 0.0))

        if node_id:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            ref = _probe_local.options(
                num_cpus=0.1,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
            ).remote(node_id, node_ip, cpus_total, gpus_total)
        else:
            ref = _probe_local.options(num_cpus=0.1).remote(
                node_id, node_ip, cpus_total, gpus_total
            )
        probe_refs.append(ref)

    results: list[NodeSnapshot] = []
    for payload in ray.get(probe_refs):
        results.append(NodeSnapshot(**payload))

    results.sort(key=lambda x: x.hostname)
    return results
