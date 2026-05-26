from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_orchestrator.models import GpuBinding, JobRequest, NodeSnapshot
from lab_orchestrator.orchestrator import Orchestrator
from lab_orchestrator.registry import ClusterRegistry, MachineSpec, save_cluster_registry


class FakeTransport:
    def __init__(self) -> None:
        self.units: dict[str, dict[str, str | None]] = {}
        self.configs: dict[str, dict[str, object]] = {}

    def start_unit(
        self,
        machine,
        unit_name: str,
        config_path: str,
        description: str,
        timeout_s: float = 15.0,
        properties=None,
    ) -> str:
        _ = machine, description, timeout_s, properties
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.configs[unit_name] = payload
        state_path = Path(payload["state_path"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "RUNNING"
        state["started_at"] = "2026-04-10T12:00:00+00:00"
        state["pid"] = 12345
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self.units[unit_name] = {"state_path": str(state_path), "active": "active"}
        return unit_name

    def show_unit(self, machine, unit_name: str, timeout_s: float = 5.0):
        _ = machine, timeout_s
        item = self.units.get(unit_name)
        if item is None or item.get("active") is None:
            return None
        return {
            "ActiveState": str(item["active"]),
            "Result": "success",
            "ExecMainStatus": "0",
        }

    def stop_unit(self, machine, unit_name: str, timeout_s: float = 10.0) -> None:
        _ = machine, timeout_s
        item = self.units[unit_name]
        state_path = Path(str(item["state_path"]))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "CANCELLED"
        state["ended_at"] = "2026-04-10T12:05:00+00:00"
        state["return_code"] = 143
        state_path.write_text(json.dumps(state), encoding="utf-8")
        item["active"] = None


class NoopStopTransport(FakeTransport):
    def stop_unit(self, machine, unit_name: str, timeout_s: float = 10.0) -> None:
        _ = machine, unit_name, timeout_s


def _node(
    host: str,
    ip: str,
    gpu_indices: list[int],
    gpus_in_use: float = 0.0,
) -> NodeSnapshot:
    return NodeSnapshot(
        node_id=host,
        ip=ip,
        hostname=host,
        cpus_total=16.0,
        gpus_total=float(len(gpu_indices)),
        cpu_percent=10.0,
        memory_total_gb=64.0,
        memory_available_gb=48.0,
        gpu_util_avg=0.0,
        gpu_memory_free_gb=32.0,
        gpus_in_use=gpus_in_use,
        gpu_users=[],
        extras={
            "gpu_details": [
                {
                    "index": idx,
                    "uuid": f"GPU-{host}-{idx}",
                    "memory_free_mib": 24000.0,
                    "util_percent": 0.0,
                    "proc_count": 0.0,
                }
                for idx in gpu_indices
            ]
        },
    )


def test_orchestrator_submit_refresh_and_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    save_cluster_registry(
        cluster_path,
        ClusterRegistry(machines=[MachineSpec(host="node33", local=True)]),
    )
    node = _node("node33", "10.0.0.33", [0, 1])
    monkeypatch.setattr(
        "lab_orchestrator.orchestrator.collect_cluster_snapshots",
        lambda registry, transport: [node],
    )

    transport = FakeTransport()
    orch = Orchestrator(
        cluster_config=str(cluster_path),
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        transport=transport,
    )
    row = orch.submit(
        JobRequest(
            name="demo",
            command="python train.py",
            cpus=2.0,
            gpus=1.0,
            workdir=str(tmp_path),
            env={},
        )
    )
    assert row["status"] == "RUNNING"
    placement = json.loads(row["placement_json"])
    assert placement["nodes"][0]["visible_gpu_indices"] == [0]

    state_path = Path(row["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "SUCCEEDED"
    state["ended_at"] = "2026-04-10T12:03:00+00:00"
    state["return_code"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    transport.units[row["remote_unit_name"]]["active"] = None

    final = orch.status(str(row["job_id"]), refresh=True)
    assert final["status"] == "SUCCEEDED"

    row2 = orch.submit(
        JobRequest(
            name="cancel-me",
            command="python hold.py",
            cpus=1.0,
            gpus=0.0,
            workdir=str(tmp_path),
            env={},
        )
    )
    cancelled = orch.cancel(str(row2["job_id"]), grace_seconds=1)
    assert cancelled["status"] == "CANCELLED"


def test_orchestrator_distributed_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    save_cluster_registry(
        cluster_path,
        ClusterRegistry(
            machines=[
                MachineSpec(host="node33", local=True),
                MachineSpec(host="node34", local=True),
            ]
        ),
    )
    nodes = [_node("node33", "10.0.0.33", [0]), _node("node34", "10.0.0.34", [0])]
    monkeypatch.setattr(
        "lab_orchestrator.orchestrator.collect_cluster_snapshots",
        lambda registry, transport: list(nodes),
    )

    transport = FakeTransport()
    orch = Orchestrator(
        cluster_config=str(cluster_path),
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        transport=transport,
    )
    row = orch.submit(
        JobRequest(
            name="dist",
            command="python ddp.py",
            cpus=4.0,
            gpus=2.0,
            distributed=True,
            workdir=str(tmp_path),
            env={},
        )
    )
    assert row["job_mode"] == "distributed"
    handles = json.loads(row["remote_handles_json"])
    assert len(handles) == 2
    env0 = transport.configs[handles[0]["unit_name"]]["env"]
    env1 = transport.configs[handles[1]["unit_name"]]["env"]
    assert env0["CUDA_VISIBLE_DEVICES"] == "0"
    assert env1["CUDA_VISIBLE_DEVICES"] == "0"
    assert env0["LOCAL_RANK"] == "0"
    assert env1["LOCAL_RANK"] == "0"
    assert env0["LAB_ORCH_VGPU_COUNT"] == "2"
    assert env1["LAB_ORCH_VGPU_COUNT"] == "2"
    assert env0["LAB_ORCH_VGPU_INDEX"] == "0"
    assert env1["LAB_ORCH_VGPU_INDEX"] == "1"
    manifest0 = json.loads(env0["LAB_ORCH_VGPU_MANIFEST_JSON"])
    manifest1 = json.loads(env1["LAB_ORCH_VGPU_MANIFEST_JSON"])
    assert [item["vgpu_index"] for item in manifest0] == [0, 1]
    assert manifest0 == manifest1


def test_orchestrator_explicit_gpu_bindings_use_node_local_visible_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    save_cluster_registry(
        cluster_path,
        ClusterRegistry(
            machines=[
                MachineSpec(host="node33", local=True),
                MachineSpec(host="node34", local=True),
            ]
        ),
    )
    nodes = [_node("node33", "10.0.0.33", [0, 1]), _node("node34", "10.0.0.34", [0, 1])]
    monkeypatch.setattr(
        "lab_orchestrator.orchestrator.collect_cluster_snapshots",
        lambda registry, transport: list(nodes),
    )

    transport = FakeTransport()
    orch = Orchestrator(
        cluster_config=str(cluster_path),
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        transport=transport,
    )
    row = orch.submit(
        JobRequest(
            name="explicit",
            command="python ddp.py",
            cpus=4.0,
            gpus=3.0,
            distributed=True,
            workdir=str(tmp_path),
            env={},
            explicit_gpu_bindings=[
                GpuBinding(host="node33", gpu_index=0),
                GpuBinding(host="node33", gpu_index=1),
                GpuBinding(host="node34", gpu_index=1),
            ],
        )
    )
    handles = json.loads(row["remote_handles_json"])
    configs = {handle["unit_name"]: transport.configs[handle["unit_name"]] for handle in handles}
    node33_envs = [cfg["env"] for cfg in configs.values() if cfg["env"]["NODE_RANK"] == "0"]
    node34_envs = [cfg["env"] for cfg in configs.values() if cfg["env"]["NODE_RANK"] == "1"]
    assert len(node33_envs) == 2
    assert len(node34_envs) == 1
    assert {env["CUDA_VISIBLE_DEVICES"] for env in node33_envs} == {"0,1"}
    assert sorted(env["LOCAL_RANK"] for env in node33_envs) == ["0", "1"]
    assert node34_envs[0]["CUDA_VISIBLE_DEVICES"] == "1"
    assert node34_envs[0]["LOCAL_RANK"] == "0"
    manifest = json.loads(node33_envs[0]["LAB_ORCH_VGPU_MANIFEST_JSON"])
    assert [item["vgpu_index"] for item in manifest] == [0, 1, 2]
    assert [item["physical_gpu"] for item in manifest] == [0, 1, 1]


def test_gpu_leases_reserve_indices_until_terminal_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    save_cluster_registry(
        cluster_path,
        ClusterRegistry(machines=[MachineSpec(host="node33", local=True)]),
    )
    node = _node("node33", "10.0.0.33", [0])
    monkeypatch.setattr(
        "lab_orchestrator.orchestrator.collect_cluster_snapshots",
        lambda registry, transport: [node],
    )

    transport = FakeTransport()
    orch = Orchestrator(
        cluster_config=str(cluster_path),
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        transport=transport,
    )
    first = orch.submit(
        JobRequest(
            name="first",
            command="python train.py",
            cpus=1.0,
            gpus=1.0,
            workdir=str(tmp_path),
            env={},
        )
    )
    assert orch.db.active_gpu_leases_by_node() == {"node33": {0}}

    with pytest.raises(RuntimeError, match="GPU capacity|requested resources"):
        orch.submit(
            JobRequest(
                name="second",
                command="python train.py",
                cpus=1.0,
                gpus=1.0,
                workdir=str(tmp_path),
                env={},
            )
        )

    state_path = Path(first["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "SUCCEEDED"
    state["ended_at"] = "2026-04-10T12:03:00+00:00"
    state["return_code"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    transport.units[first["remote_unit_name"]]["active"] = None
    orch.status(str(first["job_id"]), refresh=True)
    assert orch.db.active_gpu_leases_by_node() == {}


def test_cancel_preserves_job_that_finished_before_cancel_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    save_cluster_registry(
        cluster_path,
        ClusterRegistry(machines=[MachineSpec(host="node33", local=True)]),
    )
    node = _node("node33", "10.0.0.33", [])
    monkeypatch.setattr(
        "lab_orchestrator.orchestrator.collect_cluster_snapshots",
        lambda registry, transport: [node],
    )

    transport = NoopStopTransport()
    orch = Orchestrator(
        cluster_config=str(cluster_path),
        db_path=str(tmp_path / "jobs.db"),
        logs_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
        transport=transport,
    )
    row = orch.submit(
        JobRequest(
            name="done",
            command="python done.py",
            cpus=1.0,
            gpus=0.0,
            workdir=str(tmp_path),
            env={},
        )
    )
    state_path = Path(row["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "SUCCEEDED"
    state["ended_at"] = "2026-04-10T12:03:00+00:00"
    state["return_code"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    transport.units[row["remote_unit_name"]]["active"] = None

    cancelled = orch.cancel(str(row["job_id"]), grace_seconds=1)

    assert cancelled["status"] == "SUCCEEDED"
    assert cancelled["return_code"] == 0
