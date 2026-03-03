from __future__ import annotations

import pytest

from lab_orchestrator.models import JobRequest, NodeSnapshot
from lab_orchestrator.policy import StaticSubmissionPolicy


def _request(**kwargs) -> JobRequest:
    base = JobRequest(
        name="job",
        command="echo ok",
        cpus=4.0,
        gpus=1.0,
        workdir=".",
        env={},
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def _node(node_id: str, host: str, ip: str = "10.0.0.1") -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        ip=ip,
        hostname=host,
        cpus_total=32.0,
        gpus_total=4.0,
        cpu_percent=10.0,
        memory_total_gb=128.0,
        memory_available_gb=100.0,
        gpu_util_avg=0.0,
        gpu_memory_free_gb=24.0,
        gpus_in_use=0.0,
    )


def test_policy_enforces_active_job_limit() -> None:
    policy = StaticSubmissionPolicy(max_active_jobs_per_user=2)
    with pytest.raises(RuntimeError, match="active job limit reached"):
        policy.validate_submit(
            request=_request(),
            submit_user="alice",
            active_usage={"jobs": 2.0, "cpus": 0.0, "gpus": 0.0},
        )


def test_policy_enforces_cpu_and_gpu_limits() -> None:
    policy = StaticSubmissionPolicy(max_cpus_per_user=8.0, max_gpus_per_user=2.0)
    with pytest.raises(RuntimeError, match="CPU quota exceeded"):
        policy.validate_submit(
            request=_request(cpus=5.0, gpus=0.0),
            submit_user="alice",
            active_usage={"jobs": 1.0, "cpus": 4.0, "gpus": 0.0},
        )
    with pytest.raises(RuntimeError, match="GPU quota exceeded"):
        policy.validate_submit(
            request=_request(cpus=1.0, gpus=1.5),
            submit_user="alice",
            active_usage={"jobs": 1.0, "cpus": 1.0, "gpus": 1.0},
        )


def test_policy_filters_nodes_by_allow_and_deny_lists() -> None:
    policy = StaticSubmissionPolicy(
        allowed_hosts={"node-a", "10.0.0.2"},
        denied_hosts={"node-b"},
    )
    nodes = [
        _node("n1", "node-a", "10.0.0.1"),
        _node("n2", "node-b", "10.0.0.2"),
        _node("n3", "node-c", "10.0.0.3"),
    ]
    filtered = policy.filter_nodes(nodes)
    assert [n.hostname for n in filtered] == ["node-a"]
