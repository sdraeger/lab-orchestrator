from __future__ import annotations

import json
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

from lab_orchestrator import orchestrator as orch_mod
from lab_orchestrator.models import (
    JobRequest,
    NodeAllocation,
    NodeDecision,
    NodeSnapshot,
)


class _Policy:
    def __init__(
        self, name: str = "policy", filtered: list[NodeSnapshot] | None = None
    ):
        self.name = name
        self.filtered = filtered
        self.calls: list[tuple[str, dict[str, float]]] = []

    def validate_submit(
        self, request: JobRequest, submit_user: str, active_usage: dict[str, float]
    ) -> None:
        _ = request
        self.calls.append((submit_user, dict(active_usage)))

    def filter_nodes(self, nodes: list[NodeSnapshot]) -> list[NodeSnapshot]:
        return list(self.filtered if self.filtered is not None else nodes)


class _Strategy:
    def __init__(
        self,
        name: str = "balanced",
        single: NodeDecision | None = None,
        distributed: list[NodeAllocation] | None = None,
    ):
        self.name = name
        self.single = single
        self.distributed = distributed or []

    def select_single(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus: float,
        req_gpus: float,
        context: orch_mod.SchedulingContext,
    ) -> NodeDecision:
        _ = nodes, reservations, req_cpus, req_gpus, context
        if self.single is None:
            raise RuntimeError("single strategy not configured")
        return self.single

    def select_distributed(
        self,
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        req_cpus_total: float,
        req_gpus_total: float,
        context: orch_mod.SchedulingContext,
    ) -> list[NodeAllocation]:
        _ = nodes, reservations, req_cpus_total, req_gpus_total, context
        return list(self.distributed)


class _FakeHandle:
    def __init__(
        self,
        start_value: Any = None,
        status_value: Any = None,
        stop_value: Any = None,
        tail_value: Any = "",
    ):
        self.start = types.SimpleNamespace(remote=lambda: start_value)
        self.status = types.SimpleNamespace(remote=lambda: status_value)
        self.stop = types.SimpleNamespace(remote=lambda grace_seconds=20: stop_value)
        self.tail = types.SimpleNamespace(remote=lambda n_lines=100: tail_value)
        self.remote_kwargs: dict[str, Any] = {}


def _node(
    node_id: str = "a" * 56, host: str = "node-a", ip: str = "10.0.0.1"
) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=node_id,
        ip=ip,
        hostname=host,
        cpus_total=16.0,
        gpus_total=4.0,
        cpu_percent=10.0,
        memory_total_gb=64.0,
        memory_available_gb=48.0,
        gpu_util_avg=5.0,
        gpu_memory_free_gb=24.0,
        gpus_in_use=1.0,
    )


def _request(**overrides: Any) -> JobRequest:
    req = JobRequest(
        name="job",
        command="echo ok",
        cpus=2.0,
        gpus=1.0,
        workdir=".",
        env={"A": "1"},
        distributed=False,
        submit_user="alice",
        max_retries=1,
        retry_backoff_seconds=0.1,
        metadata={"m": "v"},
    )
    for key, value in overrides.items():
        setattr(req, key, value)
    return req


def _install_fake_ray_for_remote(
    monkeypatch: pytest.MonkeyPatch, handles: list[_FakeHandle], timeout_token: object
) -> None:
    class _RemoteBuilder:
        def options(self, **kwargs):
            self.options_kwargs = kwargs
            return self

        def remote(self, **kwargs):
            handle = handles.pop(0)
            handle.remote_kwargs = kwargs
            return handle

    def fake_remote(**kwargs):
        _ = kwargs
        return lambda cls: _RemoteBuilder()

    def fake_get(ref, timeout=None):
        _ = timeout
        if ref is timeout_token:
            raise orch_mod.GetTimeoutError("timeout")
        if ref == "RAISE":
            raise RuntimeError("boom")
        if isinstance(ref, list):
            out: list[Any] = []
            for item in ref:
                if item is timeout_token:
                    raise orch_mod.GetTimeoutError("timeout")
                if item == "RAISE":
                    raise RuntimeError("boom")
                out.append(item)
            return out
        return ref

    monkeypatch.setattr(orch_mod.ray, "remote", fake_remote)
    monkeypatch.setattr(orch_mod.ray, "get", fake_get)


def test_submit_routing_and_overview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node()
    decision = NodeDecision(
        node=node,
        score=1.0,
        cpu_free_est=8.0,
        gpu_free_est=1.0,
        reason="ok",
    )
    strategy = _Strategy(
        single=decision, distributed=[NodeAllocation(node=node, cpus=2.0, gpus=2)]
    )
    policy = _Policy()
    orch = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        placement_strategy=strategy,
        submission_policy=policy,
    )
    monkeypatch.setattr(orch_mod, "init_ray", lambda address, namespace: None)
    monkeypatch.setattr(orch_mod, "collect_cluster_snapshots", lambda: [node])
    monkeypatch.setattr(
        orch.db,
        "resource_reservations_by_node",
        lambda: {"n1": {"cpus": 0.0, "gpus": 0.0}},
    )
    monkeypatch.setattr(
        orch.db,
        "active_usage_for_user",
        lambda user: {"jobs": 0.0, "cpus": 0.0, "gpus": 0.0},
    )
    monkeypatch.setattr(
        orch.db,
        "active_usage_by_user",
        lambda: {"alice": {"jobs": 0.0, "cpus": 0.0, "gpus": 0.0}},
    )
    monkeypatch.setattr(orch.db, "active_user_reservations_by_node", lambda: {})
    monkeypatch.setattr(
        orch_mod,
        "collect_submission_metadata",
        lambda **kwargs: {"meta": kwargs["submit_user"]},
    )

    monkeypatch.setattr(
        orch, "_submit_single", lambda **kwargs: {"mode": "single", **kwargs}
    )
    monkeypatch.setattr(
        orch, "_submit_distributed", lambda **kwargs: {"mode": "distributed", **kwargs}
    )
    single = orch.submit(_request(gpus=0.5))
    assert single["mode"] == "single"
    distributed = orch.submit(_request(gpus=5.0))
    assert distributed["mode"] == "distributed"
    forced_single = orch.submit(_request(gpus=0.0, distributed=True))
    assert forced_single["mode"] == "single"

    overview = orch.overview()
    assert overview["nodes"][0].node_id == node.node_id
    assert overview["reservations"]["n1"]["cpus"] == 0.0

    policy_empty = _Policy(filtered=[])
    orch_empty = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs2.db"),
        logs_dir=str(tmp_path / "logs2"),
        placement_strategy=strategy,
        submission_policy=policy_empty,
    )
    monkeypatch.setattr(orch_mod, "init_ray", lambda address, namespace: None)
    monkeypatch.setattr(orch_mod, "collect_cluster_snapshots", lambda: [node])
    monkeypatch.setattr(
        orch_empty.db,
        "active_usage_for_user",
        lambda user: {"jobs": 0.0, "cpus": 0.0, "gpus": 0.0},
    )
    monkeypatch.setattr(orch_empty.db, "active_usage_by_user", lambda: {})
    monkeypatch.setattr(orch_empty.db, "active_user_reservations_by_node", lambda: {})
    monkeypatch.setattr(orch_empty.db, "resource_reservations_by_node", lambda: {})
    with pytest.raises(RuntimeError, match="no eligible nodes"):
        orch_empty.submit(_request())


def test_submit_single_success_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node()
    decision = NodeDecision(
        node=node, score=1.0, cpu_free_est=8.0, gpu_free_est=2.0, reason="fit"
    )
    orch = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        placement_strategy=_Strategy(single=decision),
        submission_policy=_Policy(),
    )
    req = _request()
    timeout_token = object()

    monkeypatch.setattr(
        orch_mod.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    monkeypatch.setattr(orch_mod, "utc_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[
            _FakeHandle(
                start_value={"status": "RUNNING", "started_at": 1.0, "retries_used": 1}
            )
        ],
        timeout_token=timeout_token,
    )
    row = orch._submit_single(
        request=req,
        submit_user="alice",
        metadata={"a": 1},
        nodes=[node],
        reservations={},
        context=orch_mod.SchedulingContext(submit_user="alice"),
    )
    assert row["job_id"] == "123456781234"
    assert row["status"] == "RUNNING"
    assert row["retry_attempts"] == 1

    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[_FakeHandle(start_value=timeout_token)],
        timeout_token=timeout_token,
    )
    monkeypatch.setattr(
        orch_mod.uuid,
        "uuid4",
        lambda: uuid.UUID("22345678-1234-5678-1234-567812345679"),
    )
    with pytest.raises(
        RuntimeError, match="Timed out waiting for single-node actor to start"
    ):
        orch._submit_single(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )

    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[_FakeHandle(start_value="RAISE")],
        timeout_token=timeout_token,
    )
    monkeypatch.setattr(
        orch_mod.uuid,
        "uuid4",
        lambda: uuid.UUID("32345678-1234-5678-1234-567812345680"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        orch._submit_single(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )

    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[
            _FakeHandle(
                start_value={"status": "RUNNING", "started_at": 1.0, "retries_used": 0}
            )
        ],
        timeout_token=timeout_token,
    )
    monkeypatch.setattr(
        orch_mod.uuid,
        "uuid4",
        lambda: uuid.UUID("42345678-1234-5678-1234-567812345681"),
    )
    monkeypatch.setattr(orch.db, "get_job", lambda job_id: None)
    with pytest.raises(RuntimeError, match="Could not load submitted job"):
        orch._submit_single(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )


def test_submit_distributed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node()
    req = _request(gpus=2.0, distributed=True)
    timeout_token = object()
    strategy = _Strategy(distributed=[NodeAllocation(node=node, cpus=2.0, gpus=2)])
    orch = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        placement_strategy=strategy,
        submission_policy=_Policy(),
    )
    monkeypatch.setattr(
        orch_mod.uuid,
        "uuid4",
        lambda: uuid.UUID("aaaaaaaa-1234-5678-1234-567812345678"),
    )
    monkeypatch.setattr(orch_mod.random, "randint", lambda a, b: 19000)

    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[
            _FakeHandle(
                start_value={"status": "RUNNING", "started_at": 1.0, "retries_used": 0}
            ),
            _FakeHandle(
                start_value={
                    "status": "SUCCEEDED",
                    "started_at": 2.0,
                    "ended_at": 3.0,
                    "return_code": 0,
                    "retries_used": 1,
                }
            ),
        ],
        timeout_token=timeout_token,
    )
    row = orch._submit_distributed(
        request=req,
        submit_user="alice",
        metadata={"k": "v"},
        nodes=[node],
        reservations={},
        context=orch_mod.SchedulingContext(submit_user="alice"),
    )
    assert row["job_mode"] == "distributed"
    assert "world_size=2" in row["placement_reason"]

    with pytest.raises(RuntimeError, match="no node assignments"):
        orch_mod.Orchestrator(
            ray_address="local",
            namespace="ns",
            db_path=str(tmp_path / "jobs2.db"),
            logs_dir=str(tmp_path / "logs2"),
            placement_strategy=_Strategy(distributed=[]),
            submission_policy=_Policy(),
        )._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )

    with pytest.raises(RuntimeError, match="zero workers"):
        orch_mod.Orchestrator(
            ray_address="local",
            namespace="ns",
            db_path=str(tmp_path / "jobs3.db"),
            logs_dir=str(tmp_path / "logs3"),
            placement_strategy=_Strategy(
                distributed=[NodeAllocation(node=node, cpus=1.0, gpus=0)]
            ),
            submission_policy=_Policy(),
        )._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )

    orch_timeout = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs4.db"),
        logs_dir=str(tmp_path / "logs4"),
        placement_strategy=_Strategy(
            distributed=[NodeAllocation(node=node, cpus=2.0, gpus=1)]
        ),
        submission_policy=_Policy(),
    )
    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[_FakeHandle(start_value=timeout_token)],
        timeout_token=timeout_token,
    )
    killed: list[Any] = []
    monkeypatch.setattr(
        orch_mod.ray, "kill", lambda actor, no_restart=True: killed.append(actor)
    )
    with pytest.raises(
        RuntimeError, match="Timed out waiting for distributed workers to start"
    ):
        orch_timeout._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )
    assert killed

    orch_exc = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs5.db"),
        logs_dir=str(tmp_path / "logs5"),
        placement_strategy=_Strategy(
            distributed=[NodeAllocation(node=node, cpus=2.0, gpus=1)]
        ),
        submission_policy=_Policy(),
    )
    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[_FakeHandle(start_value="RAISE")],
        timeout_token=timeout_token,
    )
    with pytest.raises(RuntimeError, match="boom"):
        orch_exc._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )


def test_orchestrator_refresh_cancel_status_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(orch, "connect", lambda: None)
    job_id = "job1"
    now = "2026-01-01T00:00:00+00:00"
    row = {
        "job_id": job_id,
        "name": "n",
        "command": "cmd",
        "requested_cpus": 1.0,
        "requested_gpus": 0.0,
        "workdir": ".",
        "env_json": "{}",
        "status": "RUNNING",
        "submit_user": "alice",
        "ray_actor_name": "actor1",
        "ray_namespace": "ns",
        "node_id": "n1",
        "node_ip": "10.0.0.1",
        "node_hostname": "host",
        "log_path": str(tmp_path / "logs" / "job1.log"),
        "created_at": now,
        "started_at": now,
        "ended_at": None,
        "return_code": None,
        "error_text": None,
        "job_mode": "single",
        "ray_actor_names_json": json.dumps(["actor1"]),
        "placement_json": json.dumps(
            {"workers": [{"log_path": str(tmp_path / "logs" / "job1.log")}]}
        ),
    }
    orch.db.insert_job(row)
    orch.db.set_job_allocations(
        job_id,
        [
            {
                "node_id": "n1",
                "node_ip": "10.0.0.1",
                "node_hostname": "host",
                "cpus": 1.0,
                "gpus": 0.0,
            }
        ],
    )

    actor = _FakeHandle(
        status_value={"status": "RUNNING", "started_at": 1.0, "retries_used": 2},
        stop_value={
            "status": "CANCELLED",
            "ended_at": 2.0,
            "return_code": 137,
            "retries_used": 2,
        },
        tail_value="live-tail\n",
    )
    monkeypatch.setattr(orch_mod.ray, "get_actor", lambda name, namespace: actor)
    monkeypatch.setattr(orch_mod.ray, "get", lambda ref, timeout=None: ref)
    monkeypatch.setattr(orch_mod, "utc_now_iso", lambda: now)

    orch.refresh_jobs()
    refreshed = orch.status(job_id, refresh=False)
    assert refreshed["status"] in {"RUNNING", "QUEUED"}
    assert refreshed["retry_attempts"] == 2

    cancelled = orch.cancel(job_id, grace_seconds=1)
    assert cancelled["status"] == "CANCELLED"

    text = orch.logs(job_id, n_lines=20)
    assert "live-tail" in text
    text_all = orch.logs_all(job_id)
    assert isinstance(text_all, str)
    src = orch.log_sources(job_id)
    assert src and src[0]["path"]

    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch.status("missing", refresh=False)
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch.cancel("missing", grace_seconds=1)
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch.logs("missing")
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch.logs_all("missing")
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch.log_sources("missing")


def test_orchestrator_helpers_and_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert orch_mod._ts_or_none(None) is None
    assert orch_mod._actor_names_from_row({"ray_actor_name": "a1"}) == ["a1"]
    assert orch_mod._actor_names_from_row({"ray_actor_names_json": '[1,"a"]'}) == [
        "1",
        "a",
    ]
    assert (
        orch_mod._actor_names_from_row(
            {"ray_actor_names_json": "bad", "ray_actor_name": ""}
        )
        == []
    )
    assert orch_mod._placement_from_row({}) == {}
    assert orch_mod._placement_from_row({"placement_json": "bad"}) == {}
    assert orch_mod._placement_from_row({"placement_json": '{"a":1}'}) == {"a": 1}

    status, *_ = orch_mod._summarize_states("QUEUED", [])
    assert status == "UNKNOWN"
    status, *_ = orch_mod._summarize_states("QUEUED", [], force_cancelled=True)
    assert status == "CANCELLED"
    assert (
        orch_mod._summarize_states("CANCELLED", [{"status": "RUNNING"}])[0]
        == "CANCELLED"
    )
    assert (
        orch_mod._summarize_states("QUEUED", [{"status": "FAILED", "return_code": 9}])[
            0
        ]
        == "FAILED"
    )
    assert orch_mod._summarize_states("QUEUED", [{"status": "RUNNING"}])[0] == "RUNNING"
    assert (
        orch_mod._summarize_states("QUEUED", [{"status": "SUCCEEDED"}])[0]
        == "SUCCEEDED"
    )
    assert orch_mod._summarize_states("QUEUED", [{"status": "QUEUED"}])[0] == "QUEUED"
    assert (
        orch_mod._summarize_states("QUEUED", [{"status": "X"}, {"status": "Y"}])[0]
        == "RUNNING"
    )
    assert orch_mod._max_retries_used([{"retries_used": "x"}, {"retries_used": 2}]) == 2

    called: list[str] = []

    class _A:
        stop = types.SimpleNamespace(
            remote=lambda grace_seconds=3: {"status": "CANCELLED"}
        )

    def fake_kill(actor, no_restart=True):
        _ = no_restart
        if not called:
            called.append("raise")
            raise RuntimeError("no kill")
        called.append("ok")

    monkeypatch.setattr(orch_mod.ray, "kill", fake_kill)
    monkeypatch.setattr(orch_mod.ray, "get", lambda ref, timeout=None: ref)
    orch_mod._stop_workers_best_effort([_A(), _A()])
    assert called == ["raise", "ok"]

    log_file = tmp_path / "x.log"
    assert orch_mod._read_log_file(log_file, 10) == ""
    log_file.write_text("a\nb\n", encoding="utf-8")
    assert orch_mod._read_log_file(log_file, None) == "a\nb\n"
    assert orch_mod._read_log_file(log_file, 1) == "b\n"

    single_sources = orch_mod._log_sources_for_row(
        {
            "job_id": "j1",
            "job_mode": "single",
            "log_path": str(log_file),
            "ray_actor_name": "a",
        },
        tmp_path,
    )
    assert single_sources[0]["label"] == "job=j1"

    distributed_row = {
        "job_id": "j2",
        "job_mode": "distributed",
        "ray_actor_names_json": json.dumps(["a1", "a2"]),
        "placement_json": json.dumps(
            {
                "workers": [
                    {"node_hostname": "h1", "log_path": str(log_file)},
                    {"node_hostname": "h2"},
                ]
            }
        ),
    }
    distributed_sources = orch_mod._log_sources_for_row(distributed_row, tmp_path)
    assert "host=h1" in distributed_sources[0]["label"]
    fallback = orch_mod._log_sources_for_row(
        {
            "job_id": "j3",
            "job_mode": "distributed",
            "ray_actor_names_json": "[]",
            "log_path": str(log_file),
        },
        tmp_path,
    )
    assert fallback[0]["label"] == "job=j3"

    assert orch_mod._max_retries_used([{}]) == 0

    class _B:
        stop = types.SimpleNamespace(remote=lambda grace_seconds=3: "RAISE")

    monkeypatch.setattr(
        orch_mod.ray,
        "kill",
        lambda actor, no_restart=True: (_ for _ in ()).throw(RuntimeError("x")),
    )
    monkeypatch.setattr(
        orch_mod.ray,
        "get",
        lambda ref, timeout=None: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    orch_mod._stop_workers_best_effort([_B()])


def test_orchestrator_additional_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node()
    req = _request(gpus=1.0, distributed=True)
    timeout_token = object()

    # Force "no worker actors created" branch.
    orch_no_actor = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-na.db"),
        logs_dir=str(tmp_path / "logs-na"),
        placement_strategy=_Strategy(
            distributed=[NodeAllocation(node=node, cpus=1.0, gpus=1)]
        ),
        submission_policy=_Policy(),
    )
    monkeypatch.setattr(orch_mod, "range", lambda n: [], raising=False)
    _install_fake_ray_for_remote(monkeypatch, handles=[], timeout_token=timeout_token)
    with pytest.raises(RuntimeError, match="no worker actors created"):
        orch_no_actor._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )
    monkeypatch.delattr(orch_mod, "range", raising=False)

    # Missing job after distributed submit path.
    orch_missing = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-miss.db"),
        logs_dir=str(tmp_path / "logs-miss"),
        placement_strategy=_Strategy(
            distributed=[NodeAllocation(node=node, cpus=1.0, gpus=1)]
        ),
        submission_policy=_Policy(),
    )
    _install_fake_ray_for_remote(
        monkeypatch,
        handles=[
            _FakeHandle(
                start_value={
                    "status": "SUCCEEDED",
                    "started_at": 1.0,
                    "ended_at": 2.0,
                    "return_code": 0,
                    "retries_used": 0,
                }
            )
        ],
        timeout_token=timeout_token,
    )
    monkeypatch.setattr(orch_missing.db, "get_job", lambda job_id: None)
    with pytest.raises(RuntimeError, match="Could not load submitted job"):
        orch_missing._submit_distributed(
            request=req,
            submit_user="alice",
            metadata={},
            nodes=[node],
            reservations={},
            context=orch_mod.SchedulingContext(submit_user="alice"),
        )

    # list_jobs refresh branch.
    orch_list = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-list.db"),
        logs_dir=str(tmp_path / "logs-list"),
    )
    calls = {"refresh": 0}
    monkeypatch.setattr(orch_list.db, "list_jobs", lambda limit=50: [{"job_id": "j"}])
    monkeypatch.setattr(
        orch_list,
        "refresh_jobs",
        lambda jobs=None: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    out = orch_list.list_jobs(limit=10, refresh=True)
    assert out == [{"job_id": "j"}]
    assert calls["refresh"] == 1

    orch_refresh = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-refresh.db"),
        logs_dir=str(tmp_path / "logs-refresh"),
    )

    # refresh_jobs branches with fake DB.
    class _DB:
        def __init__(self):
            self.updated: list[tuple[str, dict[str, Any]]] = []
            self.rows = [
                {
                    "job_id": "j-no-actor",
                    "ray_actor_name": "",
                    "ray_namespace": "ns",
                    "status": "RUNNING",
                },
                {
                    "job_id": "j-timeout",
                    "ray_actor_names_json": json.dumps(["a-timeout"]),
                    "ray_namespace": "ns",
                    "status": "RUNNING",
                },
                {
                    "job_id": "j-errors",
                    "ray_actor_names_json": json.dumps(["a-err"]),
                    "ray_namespace": "ns",
                    "status": "RUNNING",
                },
                {
                    "job_id": "j-partial",
                    "ray_actor_names_json": json.dumps(["a-ok", "a-bad"]),
                    "ray_namespace": "ns",
                    "status": "RUNNING",
                },
            ]

        def list_active_jobs(self):
            return list(self.rows)

        def update_job(self, job_id: str, **updates: Any):
            self.updated.append((job_id, updates))

    db = _DB()
    orch_refresh.db = db  # type: ignore[assignment]
    monkeypatch.setattr(orch_refresh, "connect", lambda: None)
    actors = {
        "a-timeout": _FakeHandle(status_value=timeout_token),
        "a-ok": _FakeHandle(
            status_value={"status": "RUNNING", "started_at": 1.0, "retries_used": 1}
        ),
    }

    def get_actor(name: str, namespace: str):
        _ = namespace
        if name == "a-err":
            raise RuntimeError("status bad")
        if name == "a-bad":
            raise RuntimeError("status broken")
        return actors[name]

    def fake_get(ref, timeout=None):
        _ = timeout
        if ref is timeout_token:
            raise orch_mod.GetTimeoutError("timeout")
        return ref

    monkeypatch.setattr(orch_mod.ray, "get_actor", get_actor)
    monkeypatch.setattr(orch_mod.ray, "get", fake_get)
    orch_refresh.refresh_jobs()
    updated_ids = [job_id for job_id, _ in db.updated]
    assert "j-no-actor" in updated_ids
    assert "j-errors" in updated_ids
    assert "j-partial" in updated_ids

    # status refresh -> row disappears after refresh.
    class _StatusDB:
        def __init__(self):
            self.first = True

        def get_job(self, job_id: str):
            _ = job_id
            if self.first:
                self.first = False
                return {"job_id": "x", "status": "RUNNING"}
            return None

    orch_status = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-status.db"),
        logs_dir=str(tmp_path / "logs-status"),
    )
    orch_status.db = _StatusDB()  # type: ignore[assignment]
    monkeypatch.setattr(orch_status, "refresh_jobs", lambda jobs=None: None)
    with pytest.raises(RuntimeError, match="Unknown job_id"):
        orch_status.status("x", refresh=True)

    # cancel paths: no actor names, timeout/exception errors.
    class _CancelDB:
        def __init__(self):
            self.rows = {
                "no": {"job_id": "no", "ray_actor_name": "", "ray_namespace": "ns"},
                "yes": {
                    "job_id": "yes",
                    "ray_actor_names_json": json.dumps(["c-timeout", "c-err"]),
                    "ray_namespace": "ns",
                },
            }
            self.updated: list[tuple[str, dict[str, Any]]] = []

        def get_job(self, job_id: str):
            return self.rows.get(job_id)

        def update_job(self, job_id: str, **updates: Any):
            self.updated.append((job_id, updates))

    orch_cancel = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-cancel.db"),
        logs_dir=str(tmp_path / "logs-cancel"),
    )
    cancel_db = _CancelDB()
    orch_cancel.db = cancel_db  # type: ignore[assignment]
    monkeypatch.setattr(orch_cancel, "connect", lambda: None)
    monkeypatch.setattr(
        orch_cancel,
        "status",
        lambda job_id, refresh=False: {
            "job_id": job_id,
            "status": "CANCELLED",
            "return_code": 1,
        },
    )

    result_no = orch_cancel.cancel("no", grace_seconds=1)
    assert result_no["status"] == "CANCELLED"

    def cancel_get_actor(name: str, namespace: str):
        _ = namespace
        if name == "c-timeout":
            return _FakeHandle(stop_value=timeout_token)
        raise RuntimeError("missing actor")

    monkeypatch.setattr(orch_mod.ray, "get_actor", cancel_get_actor)
    monkeypatch.setattr(
        orch_mod.ray,
        "get",
        lambda ref, timeout=None: (_ for _ in ()).throw(
            orch_mod.GetTimeoutError("timeout")
        )
        if ref is timeout_token
        else ref,
    )
    result_yes = orch_cancel.cancel("yes", grace_seconds=1)
    assert result_yes["status"] == "CANCELLED"

    # _logs_for_row branches: n_lines None multi, single fallback, distributed fallback.
    orch_logs = orch_mod.Orchestrator(
        ray_address="local",
        namespace="ns",
        db_path=str(tmp_path / "jobs-logs.db"),
        logs_dir=str(tmp_path / "logs-logs"),
    )
    monkeypatch.setattr(orch_logs, "connect", lambda: None)
    log1 = tmp_path / "l1.log"
    log2 = tmp_path / "l2.log"
    log1.write_text("one", encoding="utf-8")
    log2.write_text("two", encoding="utf-8")
    row_multi = {
        "job_id": "m1",
        "job_mode": "distributed",
        "ray_namespace": "ns",
        "ray_actor_names_json": json.dumps(["a1", "a2"]),
        "placement_json": json.dumps(
            {"workers": [{"log_path": str(log1)}, {"log_path": str(log2)}]}
        ),
    }
    text_all = orch_logs._logs_for_row(row_multi, n_lines=None)
    assert "===== rank=0" in text_all

    row_single = {
        "job_id": "s1",
        "job_mode": "single",
        "ray_namespace": "ns",
        "ray_actor_names_json": json.dumps(["a-single"]),
        "log_path": str(log1),
    }
    monkeypatch.setattr(
        orch_mod.ray,
        "get_actor",
        lambda name, namespace: (_ for _ in ()).throw(RuntimeError("no actor")),
    )
    assert orch_logs._logs_for_row(row_single, n_lines=10)

    row_dist = {
        "job_id": "d1",
        "job_mode": "distributed",
        "ray_namespace": "ns",
        "ray_actor_names_json": json.dumps(["a3", "a4"]),
        "placement_json": json.dumps(
            {"workers": [{"log_path": str(log1)}, {"log_path": str(log2)}]}
        ),
    }
    actor_ok = _FakeHandle(tail_value="tail-a")
    monkeypatch.setattr(
        orch_mod.ray,
        "get_actor",
        lambda name, namespace: actor_ok
        if name == "a3"
        else (_ for _ in ()).throw(RuntimeError("x")),
    )
    monkeypatch.setattr(orch_mod.ray, "get", lambda ref, timeout=None: ref)
    text_tail = orch_logs._logs_for_row(row_dist, n_lines=10)
    assert "tail-a" in text_tail

    monkeypatch.setattr(orch_mod, "_log_sources_for_row", lambda row, logs_dir: [])
    monkeypatch.setattr(
        orch_mod.ray,
        "get_actor",
        lambda name, namespace: (_ for _ in ()).throw(RuntimeError("no actor")),
    )
    assert (
        orch_logs._logs_for_row(
            {
                "job_id": "s2",
                "job_mode": "single",
                "ray_namespace": "ns",
                "ray_actor_names_json": json.dumps(["a"]),
            },
            n_lines=5,
        )
        == ""
    )
