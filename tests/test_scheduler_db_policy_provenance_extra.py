from __future__ import annotations

import builtins
import types
from pathlib import Path

import pytest

from lab_orchestrator import scheduler
from lab_orchestrator.db import JobDB
from lab_orchestrator.models import NodeSnapshot
from lab_orchestrator.policy import StaticSubmissionPolicy
from lab_orchestrator.policy import (
    _as_optional_float,
    _as_optional_int,
    _normalize_hosts,
    load_submission_policy,
)
from lab_orchestrator.provenance import _git_metadata, _run_git


def _node(
    node_id: str,
    host: str,
    cpus_total: float = 16.0,
    gpus_total: float = 4.0,
    cpu_percent: float = 10.0,
    gpus_in_use: float = 0.0,
) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        ip=f"10.0.0.{1 if node_id == 'n1' else 2}",
        hostname=host,
        cpus_total=cpus_total,
        gpus_total=gpus_total,
        cpu_percent=cpu_percent,
        memory_total_gb=64.0,
        memory_available_gb=32.0,
        gpu_util_avg=20.0,
        gpu_memory_free_gb=12.0,
        gpus_in_use=gpus_in_use,
    )


def test_scheduler_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    n1 = _node("n1", "node-a")
    n2 = _node("n2", "node-b", cpu_percent=80.0, gpus_in_use=3.0)
    reservations = {"n1": {"cpus": 1.0, "gpus": 0.0}, "n2": {"cpus": 0.0, "gpus": 0.0}}
    assert scheduler.format_capacity_brief(n1, reservations)[0] > 0
    assert scheduler.node_score(n1, 1.0, 1.0, req_gpus=0) != scheduler.node_score(
        n1, 1.0, 1.0, req_gpus=1
    )

    with pytest.raises(RuntimeError, match="No node satisfies"):
        scheduler.pick_best_node([n2], reservations, req_cpus=100.0, req_gpus=0.0)

    decision = scheduler.pick_best_node(
        [n1, n2],
        reservations,
        req_cpus=1.0,
        req_gpus=0.0,
        scorer=lambda node, c, g, r: 10.0 if node.node_id == "n2" else 0.0,
    )
    assert decision.node.node_id == "n2"
    with pytest.raises(RuntimeError, match="No node satisfies"):
        scheduler.pick_best_node([n2], reservations, req_cpus=1.0, req_gpus=2.0)

    with pytest.raises(RuntimeError, match="requires req_gpus_total > 0"):
        scheduler.pack_nodes_for_distributed_gpus([n1], reservations, 1.0, 0.0)
    with pytest.raises(RuntimeError, match="integer GPU count"):
        scheduler.pack_nodes_for_distributed_gpus([n1], reservations, 1.0, 1.2)
    with pytest.raises(RuntimeError, match="gpus >= 1"):
        scheduler._require_int_gpu_count(0.0)
    with pytest.raises(RuntimeError, match="No nodes have available GPU capacity"):
        scheduler.pack_nodes_for_distributed_gpus(
            [_node("n3", "node-c", gpus_total=0.0)], {}, 1.0, 1.0
        )
    with pytest.raises(RuntimeError, match="enough combined GPU capacity"):
        scheduler.pack_nodes_for_distributed_gpus(
            [_node("n4", "node-d", gpus_total=1.0)], {}, 1.0, 2.0
        )

    original_min = builtins.min

    def weird_min(a, b):
        if isinstance(a, int) and isinstance(b, int) and b == 1:
            return 0
        return original_min(a, b)

    monkeypatch.setattr(scheduler, "min", weird_min, raising=False)
    with pytest.raises(
        RuntimeError, match="Could not compute a valid distributed allocation"
    ):
        scheduler.pack_nodes_for_distributed_gpus(
            [_node("n5", "node-e", gpus_total=2.0)], {}, 2.0, 1.0
        )

    monkeypatch.setattr(scheduler, "min", original_min, raising=False)
    allocs = scheduler.pack_nodes_for_distributed_gpus(
        [_node("n6", "node-f", gpus_total=2.0)], {}, 4.0, 2.0
    )
    assert sum(a.gpus for a in allocs) == 2
    allocs_break = scheduler.pack_nodes_for_distributed_gpus(
        [
            _node("n10", "node-j", gpus_total=4.0),
            _node("n11", "node-k", gpus_total=2.0),
        ],
        {},
        2.0,
        1.0,
    )
    assert len(allocs_break) == 1
    with pytest.raises(RuntimeError, match="No nodes have available GPU capacity"):
        scheduler.pack_nodes_for_distributed_gpus(
            [_node("n12", "node-l", cpus_total=1.0, gpus_total=2.0)],
            {},
            100.0,
            2.0,
        )
    assert (
        scheduler.BalancedPlacementStrategy()
        .select_single(
            [_node("n7", "node-g")],
            {},
            req_cpus=1.0,
            req_gpus=0.0,
            context=scheduler.SchedulingContext(submit_user="alice"),
        )
        .node.node_id
        == "n7"
    )
    assert scheduler.BalancedPlacementStrategy().select_distributed(
        [_node("n8", "node-h", gpus_total=2.0)],
        {},
        req_cpus_total=2.0,
        req_gpus_total=1.0,
        context=scheduler.SchedulingContext(submit_user="alice"),
    )
    fair = scheduler.FairSharePlacementStrategy()
    context = scheduler.SchedulingContext(
        submit_user="alice",
        node_user_reservations={
            "n1": {
                "alice": {"cpus": 1.0, "gpus": 1.0},
                "bob": {"cpus": 2.0, "gpus": 1.0},
            }
        },
    )
    assert fair._score_with_fairness(
        n1, 4.0, 2.0, req_gpus=0.0, context=context
    ) <= scheduler.node_score(n1, 4.0, 2.0, req_gpus=0.0)
    assert fair.select_distributed(
        [_node("n13", "node-m", gpus_total=2.0)],
        {},
        req_cpus_total=2.0,
        req_gpus_total=1.0,
        context=context,
    )


def test_db_extra_and_policy_provenance_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "jobs.db"
    db = JobDB(db_path)
    assert db.get_job("missing") is None
    assert db.get_job_allocations("missing") == []
    db.update_job("missing")
    assert db.list_jobs(limit=10) == []
    assert db.list_active_jobs() == []
    assert db.resource_reservations_by_node() == {}
    assert db.active_usage_by_user() == {}
    assert JobDB.decode_env(JobDB.encode_env({"B": "2", "A": "1"})) == {
        "A": "1",
        "B": "2",
    }
    JobDB(db_path)

    now = "2026-01-01T00:00:00+00:00"
    row_base = {
        "name": "j",
        "command": "echo",
        "requested_cpus": 1.0,
        "requested_gpus": 1.0,
        "workdir": ".",
        "env_json": "{}",
        "status": "RUNNING",
        "submit_user": "alice",
        "ray_actor_name": "a",
        "ray_namespace": "ns",
        "node_id": "n1",
        "node_ip": "10.0.0.1",
        "node_hostname": "h1",
        "log_path": "/tmp/j.log",
        "created_at": now,
        "started_at": now,
        "ended_at": None,
        "return_code": None,
        "error_text": None,
    }
    db.insert_job({"job_id": "j1", **row_base})
    db.set_job_allocations(
        "j1",
        [
            {
                "node_id": "n1",
                "node_ip": "10.0.0.1",
                "node_hostname": "h1",
                "cpus": 1.0,
                "gpus": 1.0,
            }
        ],
    )
    db.insert_job(
        {"job_id": "j2", **row_base, "submit_user": "", "ray_actor_name": "b"}
    )
    db.set_job_allocations(
        "j2",
        [
            {
                "node_id": "n1",
                "node_ip": "10.0.0.1",
                "node_hostname": "h1",
                "cpus": 1.0,
                "gpus": 1.0,
            }
        ],
    )
    db.insert_job({"job_id": "j3", **row_base, "ray_actor_name": "c", "node_id": "n2"})
    db.insert_job(
        {
            "job_id": "j4",
            **row_base,
            "ray_actor_name": "d",
            "node_id": "",
            "submit_user": "",
        }
    )
    db.insert_job(
        {
            "job_id": "j5",
            **row_base,
            "ray_actor_name": "e",
            "node_id": "n3",
            "submit_user": "",
        }
    )
    assert db.list_jobs(limit=10)
    reservations = db.resource_reservations_by_node()
    assert reservations["n1"]["cpus"] >= 1.0
    assert reservations["n2"]["cpus"] >= 1.0
    by_user = db.active_usage_by_user()
    assert "alice" in by_user
    per_node = db.active_user_reservations_by_node()
    assert per_node["n1"]["alice"]["cpus"] >= 1.0

    class _DummyRowConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb
            return False

        def execute(self, sql, params=()):
            _ = sql, params
            return self

        def fetchone(self):
            return None

    monkeypatch.setattr(db, "_connect", lambda: _DummyRowConn())
    assert db.active_usage_for_user("alice") == {"jobs": 0.0, "cpus": 0.0, "gpus": 0.0}

    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("null\n", encoding="utf-8")
    assert load_submission_policy(str(policy_file)).name == "static"
    policy_file.write_text("- bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="policy config must be a YAML mapping"):
        load_submission_policy(str(policy_file))
    assert _normalize_hosts(None) == set()
    with pytest.raises(ValueError, match="must be a YAML list"):
        _normalize_hosts("x")
    assert _normalize_hosts([" A ", "", "b"]) == {"a", "b"}
    assert _as_optional_int("") is None
    assert _as_optional_float("") is None
    with pytest.raises(ValueError, match="must be >= 0"):
        _as_optional_int(-1)
    assert _as_optional_int("2") == 2
    with pytest.raises(ValueError, match="must be >= 0"):
        _as_optional_float(-1.0)
    policy = StaticSubmissionPolicy()
    assert policy.filter_nodes([_node("n20", "host")])[0].hostname == "host"
    assert policy.as_dict()["name"] == "static"

    assert _git_metadata(str(tmp_path / "missing")) == {"available": False}
    monkeypatch.setattr("lab_orchestrator.provenance._run_git", lambda args, cwd: None)
    assert _git_metadata(str(tmp_path)) == {"available": False}

    def fake_run_nonzero(*args, **kwargs):
        _ = args, kwargs
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("lab_orchestrator.provenance.subprocess.run", fake_run_nonzero)
    assert _run_git(["status"], cwd=tmp_path) is None

    def fake_run_raise(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr("lab_orchestrator.provenance.subprocess.run", fake_run_raise)
    assert _run_git(["status"], cwd=tmp_path) is None
