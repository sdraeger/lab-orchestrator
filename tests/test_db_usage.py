from __future__ import annotations

from pathlib import Path

from lab_orchestrator.db import JobDB
from lab_orchestrator.utils import utc_now_iso


def _row(job_id: str, submit_user: str, status: str = "RUNNING") -> dict:
    now = utc_now_iso()
    return {
        "job_id": job_id,
        "name": job_id,
        "command": "echo ok",
        "requested_cpus": 2.0,
        "requested_gpus": 1.0,
        "workdir": ".",
        "env_json": "{}",
        "status": status,
        "submit_user": submit_user,
        "ray_actor_name": f"actor-{job_id}",
        "ray_namespace": "lab-orchestrator",
        "node_id": "node-1",
        "node_ip": "10.0.0.1",
        "node_hostname": "node-1",
        "log_path": f"/tmp/{job_id}.log",
        "created_at": now,
        "started_at": now,
        "ended_at": None,
        "return_code": None,
        "error_text": None,
    }


def test_db_user_usage_and_node_reservations(tmp_path: Path) -> None:
    db = JobDB(tmp_path / "jobs.db")

    db.insert_job(_row("j1", submit_user="alice"))
    db.set_job_allocations(
        "j1",
        [
            {
                "node_id": "n1",
                "node_ip": "10.0.0.1",
                "node_hostname": "node-a",
                "cpus": 2.0,
                "gpus": 1.0,
            }
        ],
    )

    db.insert_job(_row("j2", submit_user="alice"))
    db.set_job_allocations(
        "j2",
        [
            {
                "node_id": "n2",
                "node_ip": "10.0.0.2",
                "node_hostname": "node-b",
                "cpus": 2.0,
                "gpus": 1.0,
            }
        ],
    )

    db.insert_job(_row("j3", submit_user="bob", status="QUEUED"))
    db.set_job_allocations(
        "j3",
        [
            {
                "node_id": "n1",
                "node_ip": "10.0.0.1",
                "node_hostname": "node-a",
                "cpus": 2.0,
                "gpus": 1.0,
            }
        ],
    )

    usage = db.active_usage_for_user("alice")
    assert usage["jobs"] == 2.0
    assert usage["cpus"] == 4.0
    assert usage["gpus"] == 2.0

    by_user = db.active_usage_by_user()
    assert set(by_user.keys()) == {"alice", "bob"}

    node_map = db.active_user_reservations_by_node()
    assert node_map["n1"]["alice"]["gpus"] == 1.0
    assert node_map["n1"]["bob"]["gpus"] == 1.0
