from __future__ import annotations

import math

from .models import NodeAllocation, NodeDecision, NodeSnapshot


def estimate_free_capacity(
    node: NodeSnapshot, reservations: dict[str, dict[str, float]]
) -> tuple[float, float]:
    reserved = reservations.get(node.node_id, {"cpus": 0.0, "gpus": 0.0})
    reserved_cpus = float(reserved.get("cpus", 0.0))
    reserved_gpus = float(reserved.get("gpus", 0.0))

    cpu_free_sched = max(0.0, node.cpus_total - reserved_cpus)
    cpu_free_load = max(0.0, node.cpus_total * (1.0 - node.cpu_percent / 100.0))
    cpu_free_est = min(cpu_free_sched, cpu_free_load)

    gpu_used_est = min(node.gpus_total, node.gpus_in_use + reserved_gpus)
    gpu_free_est = max(0.0, node.gpus_total - gpu_used_est)
    return cpu_free_est, gpu_free_est


def node_score(
    node: NodeSnapshot, cpu_free_est: float, gpu_free_est: float, req_gpus: float
) -> float:
    cpu_ratio = cpu_free_est / max(node.cpus_total, 1.0)
    mem_ratio = node.memory_available_gb / max(node.memory_total_gb, 1e-6)
    gpu_ratio = 1.0 if node.gpus_total <= 0 else gpu_free_est / node.gpus_total
    gpu_penalty = node.gpu_util_avg / 100.0

    if req_gpus > 0:
        return (
            0.45 * gpu_ratio + 0.30 * cpu_ratio + 0.20 * mem_ratio - 0.15 * gpu_penalty
        )
    return 0.15 * gpu_ratio + 0.50 * cpu_ratio + 0.30 * mem_ratio - 0.05 * gpu_penalty


def pick_best_node(
    nodes: list[NodeSnapshot],
    reservations: dict[str, dict[str, float]],
    req_cpus: float,
    req_gpus: float,
) -> NodeDecision:
    decisions: list[NodeDecision] = []

    for node in nodes:
        cpu_free_est, gpu_free_est = estimate_free_capacity(node, reservations)

        if cpu_free_est + 1e-9 < req_cpus:
            continue
        if gpu_free_est + 1e-9 < req_gpus:
            continue

        score = node_score(node, cpu_free_est, gpu_free_est, req_gpus=req_gpus)
        reason = (
            f"cpu_free_est={cpu_free_est:.1f}, gpu_free_est={gpu_free_est:.1f}, "
            f"cpu_load={node.cpu_percent:.1f}%, gpu_util={node.gpu_util_avg:.1f}%"
        )
        decisions.append(
            NodeDecision(
                node=node,
                score=score,
                cpu_free_est=cpu_free_est,
                gpu_free_est=gpu_free_est,
                reason=reason,
            )
        )

    if not decisions:
        raise RuntimeError(
            "No node satisfies the requested resources. "
            f"Requested cpus={req_cpus}, gpus={req_gpus}."
        )

    decisions.sort(key=lambda d: d.score, reverse=True)
    return decisions[0]


def pack_nodes_for_distributed_gpus(
    nodes: list[NodeSnapshot],
    reservations: dict[str, dict[str, float]],
    req_cpus_total: float,
    req_gpus_total: float,
) -> list[NodeAllocation]:
    if req_gpus_total <= 0:
        raise RuntimeError("Distributed GPU packing requires req_gpus_total > 0")

    req_gpus_int = _require_int_gpu_count(req_gpus_total)
    cpus_per_gpu = max(0.01, req_cpus_total / float(req_gpus_int))

    candidates: list[tuple[NodeSnapshot, float, float, float, int, int]] = []
    for node in nodes:
        cpu_free_est, gpu_free_est = estimate_free_capacity(node, reservations)
        gpu_free_int = int(math.floor(gpu_free_est + 1e-9))
        if gpu_free_int <= 0:
            continue

        gpu_limit_by_cpu = int(math.floor(cpu_free_est / cpus_per_gpu + 1e-9))
        alloc_cap = min(gpu_free_int, gpu_limit_by_cpu)
        if alloc_cap <= 0:
            continue

        score = node_score(node, cpu_free_est, gpu_free_est, req_gpus=req_gpus_total)
        candidates.append(
            (node, cpu_free_est, gpu_free_est, score, alloc_cap, gpu_free_int)
        )

    if not candidates:
        raise RuntimeError(
            "No nodes have available GPU capacity for distributed job. "
            f"Requested gpus={req_gpus_int}."
        )

    total_gpu_cap = sum(c[4] for c in candidates)
    if total_gpu_cap < req_gpus_int:
        raise RuntimeError(
            "Cluster does not currently have enough combined GPU capacity. "
            f"Requested gpus={req_gpus_int}, available_est={total_gpu_cap}."
        )

    candidates.sort(key=lambda c: (c[4], c[3]), reverse=True)

    remaining = req_gpus_int
    allocations: list[NodeAllocation] = []
    for (
        node,
        cpu_free_est,
        _gpu_free_est,
        _score,
        alloc_cap,
        _gpu_free_int,
    ) in candidates:
        if remaining <= 0:
            break
        take = min(alloc_cap, remaining)
        if take <= 0:
            continue
        alloc_cpus = min(cpu_free_est, cpus_per_gpu * float(take))
        allocations.append(NodeAllocation(node=node, cpus=alloc_cpus, gpus=int(take)))
        remaining -= take

    if remaining > 0:
        raise RuntimeError(
            "Could not compute a valid distributed allocation with current CPU/GPU load. "
            f"Unassigned GPUs={remaining}."
        )

    return allocations


def _require_int_gpu_count(value: float) -> int:
    rounded = int(round(value))
    if abs(value - float(rounded)) > 1e-6:
        raise RuntimeError(
            f"Distributed GPU jobs require an integer GPU count. Got gpus={value}."
        )
    if rounded < 1:
        raise RuntimeError(f"Distributed GPU jobs require gpus >= 1. Got gpus={value}.")
    return rounded


def format_capacity_brief(
    node: NodeSnapshot, reservations: dict[str, dict[str, float]]
) -> tuple[float, float]:
    return estimate_free_capacity(node, reservations)
