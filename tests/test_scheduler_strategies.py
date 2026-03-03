from __future__ import annotations

import pytest

from lab_orchestrator.models import NodeSnapshot
from lab_orchestrator.scheduler import (
    FairSharePlacementStrategy,
    SchedulingContext,
    resolve_placement_strategy,
)


def _node(node_id: str, host: str) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        ip=f"10.0.0.{1 if node_id == 'n1' else 2}",
        hostname=host,
        cpus_total=32.0,
        gpus_total=4.0,
        cpu_percent=10.0,
        memory_total_gb=128.0,
        memory_available_gb=120.0,
        gpu_util_avg=5.0,
        gpu_memory_free_gb=40.0,
        gpus_in_use=0.0,
    )


def test_fair_share_prefers_less_contended_node() -> None:
    nodes = [_node("n1", "node-a"), _node("n2", "node-b")]
    strategy = FairSharePlacementStrategy()
    context = SchedulingContext(
        submit_user="bob",
        node_user_reservations={
            "n1": {"alice": {"cpus": 8.0, "gpus": 2.0}},
            "n2": {},
        },
    )
    decision = strategy.select_single(
        nodes=nodes,
        reservations={},
        req_cpus=2.0,
        req_gpus=1.0,
        context=context,
    )
    assert decision.node.node_id == "n2"


def test_resolve_placement_strategy_validates_name() -> None:
    assert resolve_placement_strategy("balanced").name == "balanced"
    assert resolve_placement_strategy("fair-share").name == "fair-share"
    with pytest.raises(ValueError, match="Unknown scheduler strategy"):
        resolve_placement_strategy("unknown")
