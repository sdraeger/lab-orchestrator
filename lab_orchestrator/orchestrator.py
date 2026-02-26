from __future__ import annotations

import getpass
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray
from ray.exceptions import GetTimeoutError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .db import JobDB
from .job_actor import ScriptJobActor
from .models import JobRequest
from .probes import collect_cluster_snapshots
from .ray_runtime import init_ray
from .scheduler import (
    estimate_free_capacity,
    pack_nodes_for_distributed_gpus,
    pick_best_node,
)
from .utils import utc_now_iso

STATUS_RPC_TIMEOUT_S = 5.0
START_RPC_TIMEOUT_S = 45.0
STOP_RPC_TIMEOUT_S = 15.0
TAIL_RPC_TIMEOUT_S = 5.0


class Orchestrator:
    def __init__(
        self,
        ray_address: str,
        namespace: str,
        db_path: str,
        logs_dir: str,
    ):
        self.ray_address = ray_address
        self.namespace = namespace
        self.db = JobDB(db_path)
        self.logs_dir = Path(logs_dir).expanduser()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        init_ray(address=self.ray_address, namespace=self.namespace)

    def overview(self) -> dict[str, Any]:
        self.connect()
        nodes = collect_cluster_snapshots()
        reservations = self.db.resource_reservations_by_node()
        return {"nodes": nodes, "reservations": reservations}

    def submit(self, request: JobRequest) -> dict[str, Any]:
        self.connect()

        nodes = collect_cluster_snapshots()
        reservations = self.db.resource_reservations_by_node()
        max_single_gpu_free = max(
            (estimate_free_capacity(node, reservations)[1] for node in nodes),
            default=0.0,
        )

        distributed = bool(request.distributed)
        if not distributed and request.gpus > max_single_gpu_free + 1e-9:
            distributed = True

        if distributed and request.gpus > 0:
            return self._submit_distributed(
                request=request, nodes=nodes, reservations=reservations
            )
        return self._submit_single(
            request=request, nodes=nodes, reservations=reservations
        )

    def _submit_single(
        self,
        request: JobRequest,
        nodes: list,
        reservations: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        decision = pick_best_node(
            nodes=nodes,
            reservations=reservations,
            req_cpus=request.cpus,
            req_gpus=request.gpus,
        )

        job_id = uuid.uuid4().hex[:12]
        actor_name = f"lab-orch-job-{job_id}"
        log_path = str(self.logs_dir / f"{job_id}.log")

        row = {
            "job_id": job_id,
            "name": request.name,
            "command": request.command,
            "requested_cpus": request.cpus,
            "requested_gpus": request.gpus,
            "workdir": request.workdir,
            "env_json": JobDB.encode_env(request.env),
            "status": "QUEUED",
            "submit_user": getpass.getuser(),
            "ray_actor_name": actor_name,
            "ray_namespace": self.namespace,
            "node_id": decision.node.node_id,
            "node_ip": decision.node.ip,
            "node_hostname": decision.node.hostname,
            "log_path": log_path,
            "created_at": utc_now_iso(),
            "started_at": None,
            "ended_at": None,
            "return_code": None,
            "error_text": None,
            "job_mode": "single",
            "ray_actor_names_json": json.dumps([actor_name]),
            "placement_json": json.dumps(
                {
                    "kind": "single",
                    "reason": decision.reason,
                    "nodes": [
                        {
                            "node_id": decision.node.node_id,
                            "node_ip": decision.node.ip,
                            "node_hostname": decision.node.hostname,
                            "cpus": request.cpus,
                            "gpus": request.gpus,
                            "actor_name": actor_name,
                            "log_path": log_path,
                        }
                    ],
                }
            ),
        }
        self.db.insert_job(row)
        self.db.set_job_allocations(
            job_id,
            [
                {
                    "node_id": decision.node.node_id,
                    "node_ip": decision.node.ip,
                    "node_hostname": decision.node.hostname,
                    "cpus": request.cpus,
                    "gpus": request.gpus,
                }
            ],
        )

        remote_actor_cls = ray.remote(max_restarts=0, max_task_retries=0)(
            ScriptJobActor
        )
        actor = remote_actor_cls.options(
            name=actor_name,
            lifetime="detached",
            namespace=self.namespace,
            num_cpus=request.cpus,
            num_gpus=request.gpus,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                decision.node.node_id, soft=False
            ),
        ).remote(
            job_id=job_id,
            command=request.command,
            workdir=request.workdir,
            env=request.env,
            log_path=log_path,
        )

        try:
            state = ray.get(actor.start.remote(), timeout=START_RPC_TIMEOUT_S)
            started_at = _ts_or_none(state.get("started_at"))
            self.db.update_job(job_id, status=state["status"], started_at=started_at)
        except GetTimeoutError as exc:
            self.db.update_job(
                job_id,
                status="FAILED",
                error_text=(
                    "Timed out waiting for actor start. "
                    "Resources may have changed after scheduling."
                ),
                ended_at=utc_now_iso(),
            )
            raise RuntimeError(
                "Timed out waiting for single-node actor to start"
            ) from exc
        except Exception as exc:
            self.db.update_job(
                job_id, status="FAILED", error_text=str(exc), ended_at=utc_now_iso()
            )
            raise

        job = self.db.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Could not load submitted job {job_id}")
        job["placement_reason"] = decision.reason
        return job

    def _submit_distributed(
        self,
        request: JobRequest,
        nodes: list,
        reservations: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        allocations = pack_nodes_for_distributed_gpus(
            nodes=nodes,
            reservations=reservations,
            req_cpus_total=request.cpus,
            req_gpus_total=request.gpus,
        )
        if not allocations:
            raise RuntimeError("Distributed allocation failed: no node assignments")

        job_id = uuid.uuid4().hex[:12]
        total_workers = int(sum(a.gpus for a in allocations))
        if total_workers < 1:
            raise RuntimeError("Distributed allocation failed: zero workers")

        per_worker_cpu = max(0.01, request.cpus / float(total_workers))
        master_node = allocations[0].node
        master_addr = master_node.ip or master_node.hostname
        master_port = random.randint(18000, 32000)

        remote_actor_cls = ray.remote(max_restarts=0, max_task_retries=0)(
            ScriptJobActor
        )

        actor_names: list[str] = []
        actor_handles = []
        placement_nodes: list[dict[str, Any]] = []
        placement_workers: list[dict[str, Any]] = []
        alloc_rows: list[dict[str, Any]] = []

        global_rank = 0
        for node_rank, alloc in enumerate(allocations):
            placement_nodes.append(
                {
                    "node_id": alloc.node.node_id,
                    "node_ip": alloc.node.ip,
                    "node_hostname": alloc.node.hostname,
                    "cpus": alloc.cpus,
                    "gpus": alloc.gpus,
                }
            )
            alloc_rows.append(
                {
                    "node_id": alloc.node.node_id,
                    "node_ip": alloc.node.ip,
                    "node_hostname": alloc.node.hostname,
                    "cpus": alloc.cpus,
                    "gpus": alloc.gpus,
                }
            )

            for local_rank in range(alloc.gpus):
                actor_name = f"lab-orch-job-{job_id}-r{global_rank}"
                log_path = str(self.logs_dir / f"{job_id}.rank{global_rank}.log")
                actor_names.append(actor_name)
                placement_workers.append(
                    {
                        "rank": global_rank,
                        "local_rank": local_rank,
                        "node_rank": node_rank,
                        "node_id": alloc.node.node_id,
                        "node_ip": alloc.node.ip,
                        "node_hostname": alloc.node.hostname,
                        "actor_name": actor_name,
                        "log_path": log_path,
                    }
                )

                worker_env = dict(request.env)
                worker_env.update(
                    {
                        "WORLD_SIZE": str(total_workers),
                        "RANK": str(global_rank),
                        "LOCAL_RANK": str(local_rank),
                        "NODE_RANK": str(node_rank),
                        "LOCAL_WORLD_SIZE": str(alloc.gpus),
                        "MASTER_ADDR": str(master_addr),
                        "MASTER_PORT": str(master_port),
                        "LAB_ORCH_DISTRIBUTED": "1",
                        "LAB_ORCH_JOB_ID": job_id,
                    }
                )

                actor = remote_actor_cls.options(
                    name=actor_name,
                    lifetime="detached",
                    namespace=self.namespace,
                    num_cpus=per_worker_cpu,
                    num_gpus=1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        alloc.node.node_id, soft=False
                    ),
                ).remote(
                    job_id=job_id,
                    command=request.command,
                    workdir=request.workdir,
                    env=worker_env,
                    log_path=log_path,
                )
                actor_handles.append(actor)
                global_rank += 1

        if not actor_names:
            raise RuntimeError(
                "Distributed allocation failed: no worker actors created"
            )

        group_log_path = str(self.logs_dir / f"{job_id}.distributed.log")
        row = {
            "job_id": job_id,
            "name": request.name,
            "command": request.command,
            "requested_cpus": request.cpus,
            "requested_gpus": float(total_workers),
            "workdir": request.workdir,
            "env_json": JobDB.encode_env(request.env),
            "status": "QUEUED",
            "submit_user": getpass.getuser(),
            "ray_actor_name": actor_names[0],
            "ray_namespace": self.namespace,
            "node_id": master_node.node_id,
            "node_ip": master_node.ip,
            "node_hostname": master_node.hostname,
            "log_path": group_log_path,
            "created_at": utc_now_iso(),
            "started_at": None,
            "ended_at": None,
            "return_code": None,
            "error_text": None,
            "job_mode": "distributed",
            "ray_actor_names_json": json.dumps(actor_names),
            "placement_json": json.dumps(
                {
                    "kind": "distributed",
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "world_size": total_workers,
                    "nodes": placement_nodes,
                    "workers": placement_workers,
                }
            ),
        }
        self.db.insert_job(row)
        self.db.set_job_allocations(job_id, alloc_rows)

        try:
            start_refs = [actor.start.remote() for actor in actor_handles]
            states = ray.get(start_refs, timeout=START_RPC_TIMEOUT_S)
        except GetTimeoutError as exc:
            _stop_workers_best_effort(actor_handles)
            self.db.update_job(
                job_id,
                status="FAILED",
                error_text=(
                    "Timed out waiting for distributed workers to start. "
                    "Cluster capacity likely changed after planning."
                ),
                ended_at=utc_now_iso(),
            )
            raise RuntimeError(
                "Timed out waiting for distributed workers to start. "
                "Try lower --gpus or resubmit shortly."
            ) from exc
        except Exception as exc:
            self.db.update_job(
                job_id, status="FAILED", error_text=str(exc), ended_at=utc_now_iso()
            )
            raise

        status, started_at, ended_at, return_code = _summarize_states(
            prev_status="QUEUED",
            states=states,
        )
        self.db.update_job(
            job_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            return_code=return_code,
        )

        job = self.db.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Could not load submitted job {job_id}")
        job["placement_reason"] = (
            f"distributed allocation across {len(allocations)} nodes for world_size={total_workers}"
        )
        return job

    def list_jobs(self, limit: int = 50, refresh: bool = False) -> list[dict[str, Any]]:
        jobs = self.db.list_jobs(limit=limit)
        if refresh:
            self.refresh_jobs(jobs)
            jobs = self.db.list_jobs(limit=limit)
        return jobs

    def refresh_jobs(self, jobs: list[dict[str, Any]] | None = None) -> None:
        self.connect()
        if jobs is None:
            jobs = self.db.list_active_jobs()
        for row in jobs:
            job_id = row["job_id"]
            actor_names = _actor_names_from_row(row)
            if not actor_names:
                self.db.update_job(
                    job_id, status="UNKNOWN", error_text="no actor names recorded"
                )
                continue

            states: list[dict[str, Any]] = []
            errors: list[str] = []
            for actor_name in actor_names:
                try:
                    actor = ray.get_actor(actor_name, namespace=row["ray_namespace"])
                    states.append(
                        ray.get(actor.status.remote(), timeout=STATUS_RPC_TIMEOUT_S)
                    )
                except GetTimeoutError:
                    states.append({"status": "QUEUED"})
                    errors.append(f"{actor_name}: status timeout")
                except Exception as exc:
                    errors.append(f"{actor_name}: {exc}")

            if not states and errors:
                self.db.update_job(
                    job_id,
                    status="UNKNOWN",
                    error_text="status query failed: " + " | ".join(errors[:3]),
                )
                continue

            status, started_at, ended_at, return_code = _summarize_states(
                prev_status=str(row.get("status") or "UNKNOWN"),
                states=states,
            )

            updates: dict[str, Any] = {
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
                "return_code": return_code,
            }
            if errors:
                updates["error_text"] = "status query partial failures: " + " | ".join(
                    errors[:3]
                )
            self.db.update_job(job_id, **updates)

    def status(self, job_id: str, refresh: bool = True) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        if refresh:
            self.refresh_jobs([row])
            row = self.db.get_job(job_id)
            if row is None:
                raise RuntimeError(f"Unknown job_id '{job_id}'")
        return row

    def cancel(self, job_id: str, grace_seconds: int = 20) -> dict[str, Any]:
        self.connect()
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")

        actor_names = _actor_names_from_row(row)
        if not actor_names:
            self.db.update_job(
                job_id, status="UNKNOWN", error_text="cancel failed: no actor names"
            )
            return self.status(job_id, refresh=False)

        states: list[dict[str, Any]] = []
        errors: list[str] = []
        for actor_name in actor_names:
            try:
                actor = ray.get_actor(actor_name, namespace=row["ray_namespace"])
                states.append(
                    ray.get(
                        actor.stop.remote(grace_seconds=grace_seconds),
                        timeout=STOP_RPC_TIMEOUT_S,
                    )
                )
            except GetTimeoutError:
                errors.append(f"{actor_name}: stop timeout")
            except Exception as exc:
                errors.append(f"{actor_name}: {exc}")

        status, started_at, ended_at, return_code = _summarize_states(
            prev_status="CANCELLED",
            states=states,
            force_cancelled=True,
        )

        updates: dict[str, Any] = {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at or utc_now_iso(),
            "return_code": return_code,
        }
        if errors:
            updates["error_text"] = "cancel partial failures: " + " | ".join(errors[:3])
        self.db.update_job(job_id, **updates)
        return self.status(job_id, refresh=False)

    def logs(self, job_id: str, n_lines: int = 100) -> str:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        return self._logs_for_row(row=row, n_lines=n_lines)

    def logs_all(self, job_id: str) -> str:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        return self._logs_for_row(row=row, n_lines=None)

    def log_sources(self, job_id: str) -> list[dict[str, str]]:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        return _log_sources_for_row(row=row, logs_dir=self.logs_dir)

    def _logs_for_row(self, row: dict[str, Any], n_lines: int | None) -> str:
        actor_names = _actor_names_from_row(row)
        sources = _log_sources_for_row(row=row, logs_dir=self.logs_dir)
        is_single = (row.get("job_mode") or "single") == "single"

        if n_lines is None:
            chunks: list[str] = []
            for source in sources:
                path = Path(source["path"])
                if len(sources) > 1:
                    chunks.append(f"===== {source['label']} =====\n")
                chunks.append(_read_log_file(path, n_lines=None))
                if not chunks[-1].endswith("\n"):
                    chunks.append("\n")
            return "".join(chunks)

        if is_single and actor_names:
            try:
                self.connect()
                actor = ray.get_actor(actor_names[0], namespace=row["ray_namespace"])
                return ray.get(
                    actor.tail.remote(n_lines=n_lines), timeout=TAIL_RPC_TIMEOUT_S
                )
            except Exception:
                if sources:
                    return _read_log_file(Path(sources[0]["path"]), n_lines=n_lines)
                return ""

        chunks: list[str] = []
        for idx, source in enumerate(sources):
            actor_name = source.get("actor_name") or (
                actor_names[idx] if idx < len(actor_names) else ""
            )
            if len(sources) > 1:
                chunks.append(f"===== {source['label']} =====\n")

            tail_text = ""
            try:
                if actor_name:
                    self.connect()
                    actor = ray.get_actor(actor_name, namespace=row["ray_namespace"])
                    tail_text = ray.get(
                        actor.tail.remote(n_lines=n_lines), timeout=TAIL_RPC_TIMEOUT_S
                    )
            except Exception:
                tail_text = ""

            if not tail_text:
                tail_text = _read_log_file(Path(source["path"]), n_lines=n_lines)
            chunks.append(tail_text)
            if not tail_text.endswith("\n"):
                chunks.append("\n")
        return "".join(chunks)


def _ts_or_none(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _actor_names_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("ray_actor_names_json")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    actor = row.get("ray_actor_name")
    return [str(actor)] if actor else []


def _placement_from_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("placement_json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _summarize_states(
    prev_status: str,
    states: list[dict[str, Any]],
    force_cancelled: bool = False,
) -> tuple[str, str | None, str | None, int | None]:
    if not states:
        status = "CANCELLED" if force_cancelled else "UNKNOWN"
        return status, None, utc_now_iso() if force_cancelled else None, None

    statuses = [str(s.get("status") or "UNKNOWN") for s in states]

    started_candidates = [
        s.get("started_at") for s in states if s.get("started_at") is not None
    ]
    ended_candidates = [
        s.get("ended_at") for s in states if s.get("ended_at") is not None
    ]

    started_at = _ts_or_none(min(started_candidates)) if started_candidates else None

    if force_cancelled:
        status = "CANCELLED"
    elif prev_status == "CANCELLED":
        # Cancellation is sticky; don't resurrect a cancelled job if actor RPCs lag.
        status = "CANCELLED"
    elif "FAILED" in statuses:
        status = "FAILED"
    elif "RUNNING" in statuses:
        status = "RUNNING"
    elif all(s == "SUCCEEDED" for s in statuses):
        status = "SUCCEEDED"
    elif all(s == "QUEUED" for s in statuses):
        status = "QUEUED"
    else:
        status = "RUNNING"

    terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELLED"}
    all_terminal = all(s in terminal_statuses for s in statuses)
    ended_at = (
        _ts_or_none(max(ended_candidates))
        if ended_candidates and (all_terminal or force_cancelled)
        else None
    )

    return_code: int | None = None
    codes = [s.get("return_code") for s in states if s.get("return_code") is not None]
    if status == "SUCCEEDED":
        return_code = 0
    elif status in {"FAILED", "CANCELLED"} and codes:
        code_val = int(codes[0])
        for c in codes:
            c_int = int(c)
            if c_int != 0:
                code_val = c_int
                break
        return_code = code_val

    return status, started_at, ended_at, return_code


def _stop_workers_best_effort(worker_handles: list[Any]) -> None:
    for actor in worker_handles:
        try:
            ray.kill(actor, no_restart=True)
            continue
        except Exception:
            pass
        try:
            ray.get(actor.stop.remote(grace_seconds=3), timeout=5)
        except Exception:
            pass


def _read_log_file(path: Path, n_lines: int | None) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if n_lines is None:
            return f.read()
        lines = f.readlines()
    return "".join(lines[-max(1, int(n_lines)) :])


def _log_sources_for_row(row: dict[str, Any], logs_dir: Path) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    actor_names = _actor_names_from_row(row)
    placement = _placement_from_row(row)
    job_id = str(row["job_id"])

    if (row.get("job_mode") or "single") == "single":
        sources.append(
            {
                "label": f"job={job_id}",
                "path": str(row.get("log_path") or (logs_dir / f"{job_id}.log")),
                "actor_name": actor_names[0] if actor_names else "",
            }
        )
        return sources

    workers = placement.get("workers", []) if isinstance(placement, dict) else []
    for rank, actor_name in enumerate(actor_names):
        worker = (
            workers[rank]
            if rank < len(workers) and isinstance(workers[rank], dict)
            else {}
        )
        host = worker.get("node_hostname")
        label = f"rank={rank} actor={actor_name}"
        if host:
            label += f" host={host}"
        path = str(worker.get("log_path") or (logs_dir / f"{job_id}.rank{rank}.log"))
        sources.append({"label": label, "path": path, "actor_name": actor_name})

    if not sources:
        fallback = str(row.get("log_path") or (logs_dir / f"{job_id}.distributed.log"))
        sources.append({"label": f"job={job_id}", "path": fallback, "actor_name": ""})
    return sources
