from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .models import NodeSnapshot
from .registry import ClusterRegistry, MachineSpec
from .transport import SSHSystemdTransport


def collect_cluster_snapshots(
    registry: ClusterRegistry,
    transport: SSHSystemdTransport,
    probe_timeout_s: float = 12.0,
) -> list[NodeSnapshot]:
    machines = list(registry.machines)
    if not machines:
        return []

    results: list[NodeSnapshot] = []
    max_workers = min(max(1, len(machines)), 16)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _probe_machine,
                machine=machine,
                transport=transport,
                timeout_s=probe_timeout_s,
            ): machine
            for machine in machines
        }
        for future in as_completed(future_map):
            machine = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(_fallback_snapshot(machine, error_text=str(exc)))

    results.sort(key=lambda item: item.hostname)
    return results


def _probe_machine(
    machine: MachineSpec, transport: SSHSystemdTransport, timeout_s: float
) -> NodeSnapshot:
    payload = transport.probe_machine(machine, timeout_s=timeout_s)
    extras = dict(payload.get("extras") or {})
    extras["registered_host"] = machine.host
    extras["labels"] = list(machine.labels)
    payload["extras"] = extras
    payload["node_id"] = machine.host
    payload.setdefault("hostname", machine.host)
    payload.setdefault("ip", machine.connect_host)
    return NodeSnapshot(**payload)


def _fallback_snapshot(machine: MachineSpec, error_text: str) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=machine.host,
        ip=machine.connect_host,
        hostname=machine.host,
        cpus_total=0.0,
        gpus_total=0.0,
        cpu_percent=100.0,
        memory_total_gb=0.0,
        memory_available_gb=0.0,
        gpu_util_avg=100.0,
        gpu_memory_free_gb=0.0,
        gpus_in_use=0.0,
        gpu_users=[],
        extras={
            "probe_error": error_text,
            "registered_host": machine.host,
            "labels": list(machine.labels),
        },
    )
