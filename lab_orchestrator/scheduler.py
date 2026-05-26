from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .models import NodeAllocation, NodeDecision, NodeSnapshot

NodeScorer = Callable[[NodeSnapshot, float, float, float], float]


@dataclass(slots=True)
class SchedulingContext:
    submit_user: str
    active_user_usage: dict[str, float] = field(default_factory=dict)
    all_user_usage: dict[str, dict[str, float]] = field(default_factory=dict)
    node_user_reservations: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )


class PlacementStrategy(Protocol):
    name: str

    def select_single(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus: float,
        req_gpus: float,
        context: SchedulingContext,
    ) -> NodeDecision: ...

    def select_distributed(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus_total: float,
        req_gpus_total: float,
        context: SchedulingContext,
    ) -> list[NodeAllocation]: ...


@dataclass(slots=True)
class BalancedPlacementStrategy:
    name: str = "balanced"

    def select_single(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus: float,
        req_gpus: float,
        context: SchedulingContext,
    ) -> NodeDecision:
        _ = context
        return pick_best_node(
            nodes=nodes,
            reservations=reservations,
            req_cpus=req_cpus,
            req_gpus=req_gpus,
        )

    def select_distributed(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus_total: float,
        req_gpus_total: float,
        context: SchedulingContext,
    ) -> list[NodeAllocation]:
        _ = context
        return pack_nodes_for_distributed_gpus(
            nodes=nodes,
            reservations=reservations,
            req_cpus_total=req_cpus_total,
            req_gpus_total=req_gpus_total,
        )


@dataclass(slots=True)
class FairSharePlacementStrategy:
    # Penalize selecting nodes currently occupied by other users.
    other_user_penalty: float = 0.30
    # Penalize repeatedly selecting nodes already occupied by the submitter.
    own_user_penalty: float = 0.15
    name: str = "fair-share"

    def select_single(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus: float,
        req_gpus: float,
        context: SchedulingContext,
    ) -> NodeDecision:
        return pick_best_node(
            nodes=nodes,
            reservations=reservations,
            req_cpus=req_cpus,
            req_gpus=req_gpus,
            scorer=lambda n, c, g, rg: self._score_with_fairness(
                node=n,
                cpu_free_est=c,
                gpu_free_est=g,
                req_gpus=rg,
                context=context,
            ),
        )

    def select_distributed(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus_total: float,
        req_gpus_total: float,
        context: SchedulingContext,
    ) -> list[NodeAllocation]:
        return pack_nodes_for_distributed_gpus(
            nodes=nodes,
            reservations=reservations,
            req_cpus_total=req_cpus_total,
            req_gpus_total=req_gpus_total,
            scorer=lambda n, c, g, rg: self._score_with_fairness(
                node=n,
                cpu_free_est=c,
                gpu_free_est=g,
                req_gpus=rg,
                context=context,
            ),
        )

    def _score_with_fairness(
        self,
        node: NodeSnapshot,
        cpu_free_est: float,
        gpu_free_est: float,
        req_gpus: float,
        context: SchedulingContext,
    ) -> float:
        base = node_score(node, cpu_free_est, gpu_free_est, req_gpus=req_gpus)
        usage_by_user = context.node_user_reservations.get(node.node_id, {})

        own = usage_by_user.get(context.submit_user, {})
        own_cpus = float(own.get("cpus", 0.0))
        own_gpus = float(own.get("gpus", 0.0))
        other_cpus = 0.0
        other_gpus = 0.0
        for user, usage in usage_by_user.items():
            if user == context.submit_user:
                continue
            other_cpus += float(usage.get("cpus", 0.0))
            other_gpus += float(usage.get("gpus", 0.0))

        cpu_cap = max(node.cpus_total, 1.0)
        gpu_cap = max(node.gpus_total, 1.0)
        if req_gpus > 0:
            penalty = self.other_user_penalty * (other_gpus / gpu_cap)
            penalty += self.own_user_penalty * (own_gpus / gpu_cap)
        else:
            penalty = self.other_user_penalty * (other_cpus / cpu_cap)
            penalty += self.own_user_penalty * (own_cpus / cpu_cap)
        return base - penalty


def resolve_placement_strategy(name: str) -> PlacementStrategy:
    normalized = str(name or "").strip().lower()
    if normalized in {"fair-share", "fair_share", "fairshare"}:
        return FairSharePlacementStrategy()
    if normalized in {"balanced", ""}:
        return BalancedPlacementStrategy()
    raise ValueError(
        f"Unknown scheduler strategy '{name}'. Supported: balanced, fair-share."
    )


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
    scorer: NodeScorer | None = None,
) -> NodeDecision:
    score_fn = scorer or node_score
    decisions: list[NodeDecision] = []

    for node in nodes:
        cpu_free_est, gpu_free_est = estimate_free_capacity(node, reservations)

        if cpu_free_est + 1e-9 < req_cpus:
            continue
        if gpu_free_est + 1e-9 < req_gpus:
            continue

        score = score_fn(node, cpu_free_est, gpu_free_est, req_gpus)
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
    scorer: NodeScorer | None = None,
) -> list[NodeAllocation]:
    score_fn = scorer or node_score
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

        score = score_fn(node, cpu_free_est, gpu_free_est, req_gpus_total)
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
