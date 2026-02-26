from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NodeSnapshot:
    node_id: str
    ip: str
    hostname: str
    cpus_total: float
    gpus_total: float
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    gpu_util_avg: float
    gpu_memory_free_gb: float
    gpus_in_use: float
    gpu_users: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobRequest:
    name: str
    command: str
    cpus: float
    gpus: float
    workdir: str
    env: dict[str, str]
    distributed: bool = False


@dataclass(slots=True)
class NodeDecision:
    node: NodeSnapshot
    score: float
    cpu_free_est: float
    gpu_free_est: float
    reason: str


@dataclass(slots=True)
class NodeAllocation:
    node: NodeSnapshot
    cpus: float
    gpus: int
