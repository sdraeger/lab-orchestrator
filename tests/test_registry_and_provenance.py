from __future__ import annotations

import json
from pathlib import Path

from lab_orchestrator.db import JobDB
from lab_orchestrator.models import GpuBinding, JobRequest
from lab_orchestrator.provenance import collect_submission_metadata
from lab_orchestrator.registry import (
    ClusterRegistry,
    MachineSpec,
    load_cluster_registry,
    save_cluster_registry,
)
from lab_orchestrator.utils import utc_now_iso


def test_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cluster.yaml"
    registry = ClusterRegistry(ssh_user="alice")
    registry.upsert_machine(
        MachineSpec(host="node33", ssh_user="alice", python_bin="python3", local=True)
    )
    registry.upsert_machine(
        MachineSpec(
            host="node34",
            ssh_host="10.0.0.34",
            ssh_user="alice",
            labels=["gpu"],
        )
    )
    save_cluster_registry(path, registry)

    loaded = load_cluster_registry(path)
    assert loaded.get_machine("node33") is not None
    assert loaded.get_machine("10.0.0.34") is not None
    assert loaded.get_machine("node34").labels == ["gpu"]


def test_db_insert_and_usage(tmp_path: Path) -> None:
    db = JobDB(tmp_path / "jobs.db")
    row = {
        "job_id": "j1",
        "name": "demo",
        "command": "python train.py",
        "requested_cpus": 2.0,
        "requested_gpus": 1.0,
        "workdir": ".",
        "env_json": "{}",
        "status": "RUNNING",
        "submit_user": "alice",
        "ray_actor_name": "",
        "ray_namespace": "",
        "node_id": "node33",
        "node_ip": "10.0.0.33",
        "node_hostname": "node33",
        "log_path": "/tmp/demo.log",
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "ended_at": None,
        "return_code": None,
        "error_text": None,
        "job_mode": "single",
        "backend_name": "ssh-systemd",
        "remote_unit_name": "lab-orch-job-j1",
        "remote_handles_json": json.dumps(
            [{"host": "node33", "unit_name": "lab-orch-job-j1"}]
        ),
    }
    db.insert_job(row)
    db.set_job_allocations(
        "j1",
        [
            {
                "node_id": "node33",
                "node_ip": "10.0.0.33",
                "node_hostname": "node33",
                "cpus": 2.0,
                "gpus": 1.0,
            }
        ],
    )
    assert db.active_usage_for_user("alice") == {"jobs": 1.0, "cpus": 2.0, "gpus": 1.0}
    assert db.resource_reservations_by_node()["node33"]["gpus"] == 1.0
    db.set_gpu_leases(
        "j1",
        [
            {
                "node_id": "node33",
                "node_hostname": "node33",
                "gpu_index": 0,
                "created_at": utc_now_iso(),
            }
        ],
    )
    db.insert_job_event("j1", "unit_started", {"unit_name": "lab-orch-job-j1"})
    assert db.active_gpu_leases_by_node() == {"node33": {0}}
    assert db.list_job_events("j1")[0]["event_type"] == "unit_started"
    db.release_gpu_leases("j1")
    assert db.active_gpu_leases_by_node() == {}
    assert JobDB.decode_env(
        JobDB.encode_env({"API_TOKEN": "secret", "VISIBLE": "ok"}, redact=True)
    ) == {"API_TOKEN": "<redacted>", "VISIBLE": "ok"}


def test_collect_submission_metadata(tmp_path: Path) -> None:
    request = JobRequest(
        name="demo",
        command="python train.py",
        cpus=4.0,
        gpus=2.0,
        workdir=str(tmp_path),
        env={"A": "1"},
        distributed=True,
        submit_user="alice",
        max_retries=2,
        retry_backoff_seconds=3.0,
        metadata={"exp": "baseline"},
        explicit_gpu_bindings=[GpuBinding(host="node33", gpu_index=0)],
    )
    payload = collect_submission_metadata(
        request=request,
        scheduler_name="balanced",
        policy_name="static",
        submit_user="alice",
    )
    assert payload["backend"] == "ssh-systemd"
    assert payload["distributed"] is True
    assert payload["retry"]["max_retries"] == 2
