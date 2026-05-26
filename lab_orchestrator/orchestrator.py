from __future__ import annotations

import getpass
import json
import os
import random
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import JobDB
from .models import GpuBinding, JobRequest, NodeSnapshot
from .policy import StaticSubmissionPolicy, SubmissionPolicy
from .probes import collect_cluster_snapshots
from .provenance import collect_submission_metadata
from .registry import ClusterRegistry, MachineSpec, load_cluster_registry, save_cluster_registry
from .scheduler import (
    BalancedPlacementStrategy,
    PlacementStrategy,
    SchedulingContext,
    estimate_free_capacity,
)
from .transport import SSHSystemdTransport
from .utils import utc_now_iso

STATUS_RPC_TIMEOUT_S = 5.0
START_RPC_TIMEOUT_S = 12.0
STOP_RPC_TIMEOUT_S = 15.0
RUNNER_HEARTBEAT_INTERVAL_S = 5.0
STALE_HEARTBEAT_SECONDS = 120.0

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class Orchestrator:
    def __init__(
        self,
        cluster_config: str,
        db_path: str,
        logs_dir: str,
        state_dir: str,
        placement_strategy: PlacementStrategy | None = None,
        submission_policy: SubmissionPolicy | None = None,
        transport: SSHSystemdTransport | None = None,
    ):
        self.cluster_config = str(Path(cluster_config).expanduser())
        self.db = JobDB(db_path)
        self.logs_dir = Path(logs_dir).expanduser()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = Path(state_dir).expanduser()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.spool_dir = self.state_dir / "spool"
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.placement_strategy = placement_strategy or BalancedPlacementStrategy()
        self.submission_policy = submission_policy or StaticSubmissionPolicy()
        self.transport = transport or SSHSystemdTransport()

    def list_machines(self) -> list[MachineSpec]:
        return list(self._registry().machines)

    def register_machine(self, machine: MachineSpec) -> MachineSpec:
        registry = self._registry()
        registry.upsert_machine(machine)
        save_cluster_registry(self.cluster_config, registry)
        return machine

    def unregister_machine(self, selector: str) -> MachineSpec:
        registry = self._registry()
        removed = registry.remove_machine(selector)
        save_cluster_registry(self.cluster_config, registry)
        return removed

    def overview(self) -> dict[str, Any]:
        registry = self._registry()
        self.refresh_jobs()
        nodes = collect_cluster_snapshots(registry, self.transport)
        reservations = self.db.resource_reservations_by_node()
        gpu_index_reservations = self._active_gpu_indices_by_node()
        return {
            "nodes": nodes,
            "reservations": reservations,
            "gpu_index_reservations": gpu_index_reservations,
        }

    def doctor(self) -> list[dict[str, Any]]:
        registry = self._registry()
        rows: list[dict[str, Any]] = []
        for machine in registry.machines:
            rows.append(self.transport.doctor_machine(machine))
        rows.sort(key=lambda row: str(row.get("host") or "").lower())
        return rows

    def submit(self, request: JobRequest) -> dict[str, Any]:
        registry = self._registry()
        self.refresh_jobs()
        submit_user = str(request.submit_user or getpass.getuser())
        active_usage = self.db.active_usage_for_user(submit_user)
        self.submission_policy.validate_submit(
            request=request,
            submit_user=submit_user,
            active_usage=active_usage,
        )

        nodes = self.submission_policy.filter_nodes(
            collect_cluster_snapshots(registry, self.transport)
        )
        if not nodes:
            raise RuntimeError(
                "Submission rejected by policy: no eligible nodes after host filters."
            )

        reservations = self.db.resource_reservations_by_node()
        context = SchedulingContext(
            submit_user=submit_user,
            active_user_usage=active_usage,
            all_user_usage=self.db.active_usage_by_user(),
            node_user_reservations=self.db.active_user_reservations_by_node(),
        )
        metadata = collect_submission_metadata(
            request=request,
            scheduler_name=self.placement_strategy.name,
            policy_name=getattr(self.submission_policy, "name", ""),
            submit_user=submit_user,
        )
        if request.metadata:
            metadata["user_metadata"] = dict(request.metadata)

        if request.explicit_gpu_bindings:
            return self._submit_explicit_gpu_bindings(
                request=request,
                submit_user=submit_user,
                metadata=metadata,
                nodes=nodes,
                registry=registry,
            )

        max_single_gpu_free = max(
            (estimate_free_capacity(node, reservations)[1] for node in nodes),
            default=0.0,
        )

        distributed = bool(request.distributed)
        if not distributed and request.gpus > max_single_gpu_free + 1e-9:
            distributed = True

        if distributed and request.gpus > 0:
            return self._submit_distributed(
                request=request,
                submit_user=submit_user,
                metadata=metadata,
                nodes=nodes,
                reservations=reservations,
                context=context,
                registry=registry,
            )
        return self._submit_single(
            request=request,
            submit_user=submit_user,
            metadata=metadata,
            nodes=nodes,
            reservations=reservations,
            context=context,
            registry=registry,
        )

    def _submit_single(
        self,
        request: JobRequest,
        submit_user: str,
        metadata: dict[str, Any],
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        context: SchedulingContext,
        registry: ClusterRegistry,
    ) -> dict[str, Any]:
        decision = self.placement_strategy.select_single(
            nodes=nodes,
            reservations=reservations,
            req_cpus=request.cpus,
            req_gpus=request.gpus,
            context=context,
        )
        machine = _machine_for_node(registry, decision.node)
        if machine is None:
            raise RuntimeError(
                f"Selected node '{decision.node.hostname}' is not present in the cluster registry"
            )

        reserved_gpu_indices = self._active_gpu_indices_by_node().get(
            decision.node.node_id, set()
        )
        visible_gpu_indices = _resolve_visible_gpu_indices(
            node=decision.node,
            request=request,
            reserved_gpu_indices=reserved_gpu_indices,
        )
        env = dict(request.env)
        if visible_gpu_indices and "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in visible_gpu_indices)

        job_id = uuid.uuid4().hex[:12]
        unit_name = f"lab-orch-job-{job_id}"
        log_path = str(self.logs_dir / f"{job_id}.log")
        stderr_path = str(self.logs_dir / f"{job_id}.stderr.log")
        state_path = str(self.state_dir / f"{job_id}.json")
        config_path = str(self.spool_dir / job_id / "runner.json")
        _write_runner_config(
            config_path,
            {
                "job_id": job_id,
                "command": request.command,
                "argv": list(request.command_argv),
                "shell": bool(request.use_shell),
                "workdir": request.workdir,
                "env": env,
                "log_path": log_path,
                "stderr_path": stderr_path,
                "state_path": state_path,
                "max_retries": int(request.max_retries),
                "retry_backoff_seconds": float(request.retry_backoff_seconds),
                "heartbeat_interval_seconds": RUNNER_HEARTBEAT_INTERVAL_S,
            },
        )
        _write_runner_state_placeholder(state_path, job_id=job_id)

        handle = {
            "host": machine.host,
            "unit_name": unit_name,
            "state_path": state_path,
            "log_path": log_path,
            "stdout_path": log_path,
            "stderr_path": stderr_path,
            "config_path": config_path,
            "node_id": decision.node.node_id,
            "node_ip": decision.node.ip,
            "node_hostname": decision.node.hostname,
            "visible_gpu_indices": visible_gpu_indices,
            "local": bool(machine.local),
        }

        row = {
            "job_id": job_id,
            "name": request.name,
            "command": request.command,
            "requested_cpus": request.cpus,
            "requested_gpus": request.gpus,
            "workdir": request.workdir,
            "env_json": JobDB.encode_env(env, redact=True),
            "status": "QUEUED",
            "submit_user": submit_user,
            "ray_actor_name": "",
            "ray_namespace": "",
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
            "ray_actor_names_json": None,
            "scheduler_name": self.placement_strategy.name,
            "policy_name": getattr(self.submission_policy, "name", ""),
            "retry_max": int(request.max_retries),
            "retry_backoff_seconds": float(request.retry_backoff_seconds),
            "retry_attempts": 0,
            "metadata_json": json.dumps(metadata, sort_keys=True),
            "placement_json": json.dumps(
                {
                    "kind": "single",
                    "strategy": self.placement_strategy.name,
                    "policy": getattr(self.submission_policy, "name", ""),
                    "reason": decision.reason,
                    "nodes": [
                        {
                            "node_id": decision.node.node_id,
                            "node_ip": decision.node.ip,
                            "node_hostname": decision.node.hostname,
                            "cpus": request.cpus,
                            "gpus": request.gpus,
                            "visible_gpu_indices": visible_gpu_indices,
                            "unit_name": unit_name,
                            "state_path": state_path,
                            "log_path": log_path,
                            "stdout_path": log_path,
                            "stderr_path": stderr_path,
                        }
                    ],
                    "workers": [],
                }
            ),
            "backend_name": "ssh-systemd",
            "remote_unit_name": unit_name,
            "remote_handles_json": json.dumps([handle]),
            "state_path": state_path,
            "spool_path": config_path,
        }
        alloc_rows = [
            {
                "node_id": decision.node.node_id,
                "node_ip": decision.node.ip,
                "node_hostname": decision.node.hostname,
                "cpus": request.cpus,
                "gpus": request.gpus,
            }
        ]
        leases = _gpu_leases_for_node(
            job_id=job_id,
            node_id=decision.node.node_id,
            node_hostname=decision.node.hostname,
            gpu_indices=visible_gpu_indices,
        )
        with self.db.write_transaction() as con:
            self.db.insert_job(row, con=con)
            self.db.set_job_allocations(job_id, alloc_rows, con=con)
            self.db.set_gpu_leases(job_id, leases, con=con)
            self.db.insert_job_event(
                job_id,
                "submitted",
                {"mode": "single", "node": decision.node.hostname},
                con=con,
            )
            self.db.insert_job_event(
                job_id,
                "allocated",
                {"allocations": alloc_rows, "gpu_leases": leases},
                con=con,
            )

        try:
            self.transport.start_unit(
                machine=machine,
                unit_name=unit_name,
                config_path=config_path,
                description=f"lab-orch {job_id} {request.name}",
                timeout_s=START_RPC_TIMEOUT_S,
                properties=request.systemd_properties,
            )
            self.db.insert_job_event(
                job_id,
                "unit_started",
                {"unit_name": unit_name, "host": machine.host},
            )
        except Exception as exc:
            self.db.update_job(
                job_id,
                status="FAILED",
                error_text=str(exc),
                ended_at=utc_now_iso(),
            )
            self.db.release_gpu_leases(job_id)
            self.db.insert_job_event(
                job_id,
                "unit_start_failed",
                {"unit_name": unit_name, "host": machine.host, "error": str(exc)},
            )
            raise

        self._refresh_job_by_id(job_id)
        job = self.db.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Could not load submitted job {job_id}")
        job["placement_reason"] = decision.reason
        return job

    def _submit_distributed(
        self,
        request: JobRequest,
        submit_user: str,
        metadata: dict[str, Any],
        nodes: list[NodeSnapshot],
        reservations: dict[str, dict[str, float]],
        context: SchedulingContext,
        registry: ClusterRegistry,
    ) -> dict[str, Any]:
        allocations = self.placement_strategy.select_distributed(
            nodes=nodes,
            reservations=reservations,
            req_cpus_total=request.cpus,
            req_gpus_total=request.gpus,
            context=context,
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

        reserved_gpu_indices = self._active_gpu_indices_by_node()
        placement_nodes: list[dict[str, Any]] = []
        placement_workers: list[dict[str, Any]] = []
        alloc_rows: list[dict[str, Any]] = []
        handles: list[dict[str, Any]] = []
        config_paths_by_unit: dict[str, str] = {}
        visible_specs_by_unit: dict[str, str] = {}

        global_rank = 0
        for node_rank, alloc in enumerate(allocations):
            machine = _machine_for_node(registry, alloc.node)
            if machine is None:
                raise RuntimeError(
                    f"Selected node '{alloc.node.hostname}' is not present in the cluster registry"
                )
            gpu_indices = _auto_select_gpu_indices(
                node=alloc.node,
                count=int(alloc.gpus),
                reserved_gpu_indices=reserved_gpu_indices.get(alloc.node.node_id, set()),
            )
            reserved_gpu_indices.setdefault(alloc.node.node_id, set()).update(gpu_indices)
            visible_gpu_spec = ",".join(str(int(idx)) for idx in gpu_indices)

            placement_nodes.append(
                {
                    "node_id": alloc.node.node_id,
                    "node_ip": alloc.node.ip,
                    "node_hostname": alloc.node.hostname,
                    "cpus": alloc.cpus,
                    "gpus": alloc.gpus,
                    "visible_gpu_indices": gpu_indices,
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

            for local_rank, physical_gpu in enumerate(gpu_indices):
                unit_name = f"lab-orch-job-{job_id}-r{global_rank}"
                log_path = str(self.logs_dir / f"{job_id}.rank{global_rank}.log")
                stderr_path = str(self.logs_dir / f"{job_id}.rank{global_rank}.stderr.log")
                state_path = str(self.state_dir / f"{job_id}.rank{global_rank}.json")
                config_path = str(
                    self.spool_dir / job_id / f"runner.rank{global_rank}.json"
                )
                worker_env = dict(request.env)
                worker_env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": visible_gpu_spec,
                        "WORLD_SIZE": str(total_workers),
                        "RANK": str(global_rank),
                        "LOCAL_RANK": str(local_rank),
                        "NODE_RANK": str(node_rank),
                        "LOCAL_WORLD_SIZE": str(alloc.gpus),
                        "MASTER_ADDR": str(master_addr),
                        "MASTER_PORT": str(master_port),
                        "LAB_ORCH_DISTRIBUTED": "1",
                        "LAB_ORCH_JOB_ID": job_id,
                        "LAB_ORCH_PHYSICAL_GPU": str(int(physical_gpu)),
                    }
                )
                _write_runner_config(
                    config_path,
                    {
                        "job_id": job_id,
                        "command": request.command,
                        "argv": list(request.command_argv),
                        "shell": bool(request.use_shell),
                        "workdir": request.workdir,
                        "env": worker_env,
                        "log_path": log_path,
                        "stderr_path": stderr_path,
                        "state_path": state_path,
                        "max_retries": int(request.max_retries),
                        "retry_backoff_seconds": float(request.retry_backoff_seconds),
                        "heartbeat_interval_seconds": RUNNER_HEARTBEAT_INTERVAL_S,
                    },
                )
                config_paths_by_unit[unit_name] = config_path
                visible_specs_by_unit[unit_name] = visible_gpu_spec
                _write_runner_state_placeholder(state_path, job_id=job_id)

                handle = {
                    "host": machine.host,
                    "unit_name": unit_name,
                    "state_path": state_path,
                    "log_path": log_path,
                    "stdout_path": log_path,
                    "stderr_path": stderr_path,
                    "config_path": config_path,
                    "node_id": alloc.node.node_id,
                    "node_ip": alloc.node.ip,
                    "node_hostname": alloc.node.hostname,
                    "physical_gpu": int(physical_gpu),
                    "rank": global_rank,
                    "local_rank": local_rank,
                    "node_rank": node_rank,
                    "local": bool(machine.local),
                }
                handles.append(handle)
                placement_workers.append({**handle, "virtual_gpu": global_rank})
                global_rank += 1

        vgpu_manifest = _build_vgpu_manifest(
            handles=handles,
            visible_specs_by_unit=visible_specs_by_unit,
        )
        _inject_vgpu_envs_into_runner_configs(
            handles=handles,
            config_paths_by_unit=config_paths_by_unit,
            visible_specs_by_unit=visible_specs_by_unit,
            vgpu_manifest=vgpu_manifest,
        )

        group_log_path = str(self.logs_dir / f"{job_id}.distributed.log")
        row = {
            "job_id": job_id,
            "name": request.name,
            "command": request.command,
            "requested_cpus": request.cpus,
            "requested_gpus": float(total_workers),
            "workdir": request.workdir,
            "env_json": JobDB.encode_env(request.env, redact=True),
            "status": "QUEUED",
            "submit_user": submit_user,
            "ray_actor_name": "",
            "ray_namespace": "",
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
            "ray_actor_names_json": None,
            "scheduler_name": self.placement_strategy.name,
            "policy_name": getattr(self.submission_policy, "name", ""),
            "retry_max": int(request.max_retries),
            "retry_backoff_seconds": float(request.retry_backoff_seconds),
            "retry_attempts": 0,
            "metadata_json": json.dumps(metadata, sort_keys=True),
            "placement_json": json.dumps(
                {
                    "kind": "distributed",
                    "strategy": self.placement_strategy.name,
                    "policy": getattr(self.submission_policy, "name", ""),
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "world_size": total_workers,
                    "nodes": placement_nodes,
                    "workers": placement_workers,
                }
            ),
            "backend_name": "ssh-systemd",
            "remote_unit_name": handles[0]["unit_name"],
            "remote_handles_json": json.dumps(handles),
            "state_path": handles[0]["state_path"],
            "spool_path": str(self.spool_dir / job_id),
        }
        leases = _gpu_leases_for_placement_nodes(job_id, placement_nodes)
        with self.db.write_transaction() as con:
            self.db.insert_job(row, con=con)
            self.db.set_job_allocations(job_id, alloc_rows, con=con)
            self.db.set_gpu_leases(job_id, leases, con=con)
            self.db.insert_job_event(
                job_id,
                "submitted",
                {"mode": "distributed", "world_size": total_workers},
                con=con,
            )
            self.db.insert_job_event(
                job_id,
                "allocated",
                {"allocations": alloc_rows, "gpu_leases": leases},
                con=con,
            )

        try:
            self._start_handles_parallel(handles, registry, job_id, request)
        except Exception as exc:
            self.db.update_job(
                job_id,
                status="FAILED",
                error_text=str(exc),
                ended_at=utc_now_iso(),
            )
            self.db.release_gpu_leases(job_id)
            self._stop_handles_best_effort(handles, registry)
            raise

        self._refresh_job_by_id(job_id)
        job = self.db.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Could not load submitted job {job_id}")
        job["placement_reason"] = (
            f"distributed allocation across {len(allocations)} nodes for world_size={total_workers}"
        )
        return job

    def _submit_explicit_gpu_bindings(
        self,
        request: JobRequest,
        submit_user: str,
        metadata: dict[str, Any],
        nodes: list[NodeSnapshot],
        registry: ClusterRegistry,
    ) -> dict[str, Any]:
        bindings = list(request.explicit_gpu_bindings)
        if not bindings:
            raise RuntimeError(
                "Explicit GPU binding submit requires at least one binding"
            )

        node_lookup: dict[str, NodeSnapshot] = {}
        for node in nodes:
            for key in {
                str(node.hostname or "").strip().lower(),
                str(node.ip or "").strip().lower(),
                str(node.node_id or "").strip().lower(),
            }:
                if key:
                    node_lookup.setdefault(key, node)

        reserved_gpu_indices = self._active_gpu_indices_by_node()
        workers: list[tuple[NodeSnapshot, GpuBinding]] = []
        seen_pairs: set[tuple[str, int]] = set()
        for binding in bindings:
            node = node_lookup.get(str(binding.host).strip().lower())
            if node is None:
                raise RuntimeError(
                    f"Explicit GPU binding host '{binding.host}' is not available after policy filtering."
                )
            max_gpu = int(node.gpus_total)
            gpu_idx = int(binding.gpu_index)
            if gpu_idx < 0 or gpu_idx >= max_gpu:
                raise RuntimeError(
                    f"Explicit GPU binding '{binding.host}:{gpu_idx}' is out of range for node "
                    f"{node.hostname} (available GPU indices: 0..{max(0, max_gpu - 1)})."
                )
            pair = (str(node.node_id), gpu_idx)
            if pair in seen_pairs:
                raise RuntimeError(
                    f"Duplicate explicit GPU binding detected for node={node.hostname}, gpu={gpu_idx}."
                )
            if gpu_idx in reserved_gpu_indices.get(str(node.node_id), set()):
                raise RuntimeError(
                    f"GPU {node.hostname}:{gpu_idx} is already reserved by another active job."
                )
            seen_pairs.add(pair)
            reserved_gpu_indices.setdefault(str(node.node_id), set()).add(gpu_idx)
            workers.append((node, binding))

        total_workers = len(workers)
        if request.gpus > 0 and abs(float(request.gpus) - float(total_workers)) > 1e-9:
            raise RuntimeError(
                f"Explicit GPU binding count ({total_workers}) does not match requested gpus={request.gpus:g}."
            )

        job_id = uuid.uuid4().hex[:12]
        master_node = workers[0][0]
        master_addr = master_node.ip or master_node.hostname
        master_port = random.randint(18000, 32000)
        per_worker_cpu = max(0.01, request.cpus / float(total_workers))

        handle_rows: list[dict[str, Any]] = []
        placement_workers: list[dict[str, Any]] = []
        placement_nodes: dict[str, dict[str, Any]] = {}
        rank_per_node: dict[str, int] = {}
        local_world_size_by_node: dict[str, int] = {}
        physical_gpus_by_node: dict[str, list[int]] = {}
        config_paths_by_unit: dict[str, str] = {}
        visible_specs_by_unit: dict[str, str] = {}
        for node, _binding in workers:
            local_world_size_by_node[str(node.node_id)] = (
                local_world_size_by_node.get(str(node.node_id), 0) + 1
            )
        for node, binding in workers:
            physical_gpus_by_node.setdefault(str(node.node_id), []).append(
                int(binding.gpu_index)
            )
        visible_gpu_spec_by_node = {
            node_key: ",".join(str(idx) for idx in gpu_indices)
            for node_key, gpu_indices in physical_gpus_by_node.items()
        }

        for rank, (node, binding) in enumerate(workers):
            machine = _machine_for_node(registry, node)
            if machine is None:
                raise RuntimeError(f"Unknown registered machine for node '{node.hostname}'")
            node_key = str(node.node_id)
            local_rank = rank_per_node.get(node_key, 0)
            rank_per_node[node_key] = local_rank + 1
            node_rank = list(local_world_size_by_node.keys()).index(node_key)
            unit_name = f"lab-orch-job-{job_id}-r{rank}"
            log_path = str(self.logs_dir / f"{job_id}.rank{rank}.log")
            stderr_path = str(self.logs_dir / f"{job_id}.rank{rank}.stderr.log")
            state_path = str(self.state_dir / f"{job_id}.rank{rank}.json")
            config_path = str(self.spool_dir / job_id / f"runner.rank{rank}.json")
            worker_env = dict(request.env)
            worker_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": visible_gpu_spec_by_node[node_key],
                    "WORLD_SIZE": str(total_workers),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(local_rank),
                    "NODE_RANK": str(node_rank),
                    "LOCAL_WORLD_SIZE": str(local_world_size_by_node[node_key]),
                    "MASTER_ADDR": str(master_addr),
                    "MASTER_PORT": str(master_port),
                    "LAB_ORCH_DISTRIBUTED": "1",
                    "LAB_ORCH_JOB_ID": job_id,
                    "LAB_ORCH_VGPU": "1",
                    "LAB_ORCH_VGPU_INDEX": str(rank),
                    "LAB_ORCH_VGPU_COUNT": str(total_workers),
                    "LAB_ORCH_VGPU_HOST": str(node.hostname),
                    "LAB_ORCH_VGPU_PHYSICAL_GPU": str(int(binding.gpu_index)),
                    "LAB_ORCH_PHYSICAL_GPU": str(int(binding.gpu_index)),
                }
            )
            _write_runner_config(
                config_path,
                {
                    "job_id": job_id,
                    "command": request.command,
                    "argv": list(request.command_argv),
                    "shell": bool(request.use_shell),
                    "workdir": request.workdir,
                    "env": worker_env,
                    "log_path": log_path,
                    "stderr_path": stderr_path,
                    "state_path": state_path,
                    "max_retries": int(request.max_retries),
                    "retry_backoff_seconds": float(request.retry_backoff_seconds),
                    "heartbeat_interval_seconds": RUNNER_HEARTBEAT_INTERVAL_S,
                },
            )
            config_paths_by_unit[unit_name] = config_path
            visible_specs_by_unit[unit_name] = visible_gpu_spec_by_node[node_key]
            _write_runner_state_placeholder(state_path, job_id=job_id)
            handle = {
                "host": machine.host,
                "unit_name": unit_name,
                "state_path": state_path,
                "log_path": log_path,
                "stdout_path": log_path,
                "stderr_path": stderr_path,
                "config_path": config_path,
                "node_id": node.node_id,
                "node_ip": node.ip,
                "node_hostname": node.hostname,
                "physical_gpu": int(binding.gpu_index),
                "rank": rank,
                "local_rank": local_rank,
                "node_rank": node_rank,
                "local": bool(machine.local),
            }
            handle_rows.append(handle)
            placement_workers.append({**handle, "virtual_gpu": rank})
            entry = placement_nodes.setdefault(
                node_key,
                {
                    "node_id": node.node_id,
                    "node_ip": node.ip,
                    "node_hostname": node.hostname,
                    "cpus": 0.0,
                    "gpus": 0.0,
                    "visible_gpu_indices": [],
                },
            )
            entry["cpus"] += per_worker_cpu
            entry["gpus"] += 1.0
            entry["visible_gpu_indices"].append(int(binding.gpu_index))

        vgpu_manifest = _build_vgpu_manifest(
            handles=handle_rows,
            visible_specs_by_unit=visible_specs_by_unit,
        )
        _inject_vgpu_envs_into_runner_configs(
            handles=handle_rows,
            config_paths_by_unit=config_paths_by_unit,
            visible_specs_by_unit=visible_specs_by_unit,
            vgpu_manifest=vgpu_manifest,
        )

        group_log_path = str(self.logs_dir / f"{job_id}.distributed.log")
        row = {
            "job_id": job_id,
            "name": request.name,
            "command": request.command,
            "requested_cpus": request.cpus,
            "requested_gpus": float(total_workers),
            "workdir": request.workdir,
            "env_json": JobDB.encode_env(request.env, redact=True),
            "status": "QUEUED",
            "submit_user": submit_user,
            "ray_actor_name": "",
            "ray_namespace": "",
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
            "ray_actor_names_json": None,
            "scheduler_name": self.placement_strategy.name,
            "policy_name": getattr(self.submission_policy, "name", ""),
            "retry_max": int(request.max_retries),
            "retry_backoff_seconds": float(request.retry_backoff_seconds),
            "retry_attempts": 0,
            "metadata_json": json.dumps(metadata, sort_keys=True),
            "placement_json": json.dumps(
                {
                    "kind": "explicit-gpu-bindings",
                    "strategy": self.placement_strategy.name,
                    "policy": getattr(self.submission_policy, "name", ""),
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "world_size": total_workers,
                    "nodes": list(placement_nodes.values()),
                    "workers": placement_workers,
                }
            ),
            "backend_name": "ssh-systemd",
            "remote_unit_name": handle_rows[0]["unit_name"],
            "remote_handles_json": json.dumps(handle_rows),
            "state_path": handle_rows[0]["state_path"],
            "spool_path": str(self.spool_dir / job_id),
        }
        alloc_rows = list(placement_nodes.values())
        leases = _gpu_leases_for_placement_nodes(job_id, alloc_rows)
        with self.db.write_transaction() as con:
            self.db.insert_job(row, con=con)
            self.db.set_job_allocations(job_id, alloc_rows, con=con)
            self.db.set_gpu_leases(job_id, leases, con=con)
            self.db.insert_job_event(
                job_id,
                "submitted",
                {"mode": "explicit-gpu-bindings", "world_size": total_workers},
                con=con,
            )
            self.db.insert_job_event(
                job_id,
                "allocated",
                {"allocations": alloc_rows, "gpu_leases": leases},
                con=con,
            )

        try:
            self._start_handles_parallel(handle_rows, registry, job_id, request)
        except Exception as exc:
            self.db.update_job(
                job_id,
                status="FAILED",
                error_text=str(exc),
                ended_at=utc_now_iso(),
            )
            self.db.release_gpu_leases(job_id)
            self._stop_handles_best_effort(handle_rows, registry)
            raise

        self._refresh_job_by_id(job_id)
        job = self.db.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Could not load submitted job {job_id}")
        job["placement_reason"] = (
            f"explicit gpu map across {len(placement_nodes)} nodes for world_size={total_workers}"
        )
        return job

    def list_jobs(self, limit: int = 50, refresh: bool = False) -> list[dict[str, Any]]:
        jobs = self.db.list_jobs(limit=limit)
        if refresh:
            self.refresh_jobs(jobs)
            jobs = self.db.list_jobs(limit=limit)
        return jobs

    def refresh_jobs(self, jobs: list[dict[str, Any]] | None = None) -> None:
        if jobs is None:
            jobs = self.db.list_active_jobs()
        for row in jobs:
            self._refresh_job_row(row)

    def status(self, job_id: str, refresh: bool = True) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        if refresh:
            self._refresh_job_row(row)
            row = self.db.get_job(job_id)
            if row is None:
                raise RuntimeError(f"Unknown job_id '{job_id}'")
        return row

    def cancel(self, job_id: str, grace_seconds: int = 20) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")

        handles = _remote_handles_from_row(row)
        if not handles:
            self.db.update_job(
                job_id, status="UNKNOWN", error_text="cancel failed: no remote handles"
            )
            return self.status(job_id, refresh=False)

        registry = self._registry()
        errors: list[str] = []
        for handle in handles:
            machine = registry.get_machine(str(handle.get("host") or ""))
            if machine is None:
                errors.append(f"{handle.get('unit_name')}: unknown machine")
                continue
            try:
                self.transport.stop_unit(
                    machine=machine,
                    unit_name=str(handle["unit_name"]),
                    timeout_s=min(float(grace_seconds), STOP_RPC_TIMEOUT_S),
                )
            except Exception as exc:
                errors.append(f"{handle.get('unit_name')}: {exc}")

        deadline = time.time() + max(1.0, float(grace_seconds))
        while time.time() < deadline:
            self._refresh_job_by_id(job_id)
            refreshed = self.db.get_job(job_id)
            if refreshed is None:
                break
            if str(refreshed.get("status")) in TERMINAL_STATUSES:
                break
            time.sleep(0.25)

        refreshed = self.db.get_job(job_id)
        if refreshed is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")

        current_status = str(refreshed.get("status") or "")
        if current_status in TERMINAL_STATUSES and current_status != "CANCELLED":
            if errors:
                self.db.update_job(
                    job_id,
                    error_text="cancel partial failures after terminal state: "
                    + " | ".join(errors[:3]),
                )
            return self.status(job_id, refresh=False)

        updates: dict[str, Any] = {
            "status": "CANCELLED",
            "ended_at": refreshed.get("ended_at") or utc_now_iso(),
        }
        if refreshed.get("return_code") is None:
            updates["return_code"] = 143
        if errors:
            updates["error_text"] = "cancel partial failures: " + " | ".join(errors[:3])
        self.db.update_job(job_id, **updates)
        self.db.release_gpu_leases(job_id)
        self.db.insert_job_event(
            job_id,
            "cancelled",
            {"return_code": updates.get("return_code", refreshed.get("return_code"))},
        )
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
        sources = _log_sources_for_row(row=row, logs_dir=self.logs_dir)
        chunks: list[str] = []
        visible: list[tuple[dict[str, str], str]] = []
        for source in sources:
            chunk = _read_log_file(Path(source["path"]), n_lines=n_lines)
            if chunk:
                visible.append((source, chunk))
        if not visible:
            return ""
        for source, chunk in visible:
            if len(visible) > 1:
                chunks.append(f"===== {source['label']} =====\n")
            chunks.append(chunk)
            if not chunks[-1].endswith("\n"):
                chunks.append("\n")
        return "".join(chunks)

    def _refresh_job_by_id(self, job_id: str) -> None:
        row = self.db.get_job(job_id)
        if row is None:
            raise RuntimeError(f"Unknown job_id '{job_id}'")
        self._refresh_job_row(row)

    def _refresh_job_row(self, row: dict[str, Any]) -> None:
        job_id = str(row["job_id"])
        handles = _remote_handles_from_row(row)
        if not handles:
            self.db.update_job(
                job_id, status="UNKNOWN", error_text="no remote handles recorded"
            )
            return

        registry = self._registry()
        states: list[dict[str, Any]] = []
        errors: list[str] = []
        for handle in handles:
            machine = registry.get_machine(str(handle.get("host") or ""))
            if machine is None:
                errors.append(f"{handle.get('unit_name')}: unknown machine")
                continue
            try:
                states.append(self._state_for_handle(machine, handle))
            except Exception as exc:
                errors.append(f"{handle.get('unit_name')}: {exc}")

        if not states and errors:
            self.db.update_job(
                job_id,
                status="UNKNOWN",
                error_text="status query failed: " + " | ".join(errors[:3]),
            )
            return

        status, started_at, ended_at, return_code = _summarize_states(
            prev_status=str(row.get("status") or "UNKNOWN"),
            states=states,
        )
        updates: dict[str, Any] = {
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "return_code": return_code,
            "retry_attempts": _max_retries_used(states),
        }
        if errors:
            updates["error_text"] = "status query partial failures: " + " | ".join(
                errors[:3]
            )
        self.db.update_job(job_id, **updates)
        if status in TERMINAL_STATUSES:
            self.db.release_gpu_leases(job_id)
        if status != str(row.get("status") or ""):
            self.db.insert_job_event(
                job_id,
                "status_changed",
                {
                    "previous_status": str(row.get("status") or ""),
                    "status": status,
                    "return_code": return_code,
                },
            )

    def _state_for_handle(
        self, machine: MachineSpec, handle: dict[str, Any]
    ) -> dict[str, Any]:
        state = _load_runner_state(str(handle.get("state_path") or ""))
        if state and str(state.get("status")) in TERMINAL_STATUSES:
            return state

        unit_name = str(handle.get("unit_name") or "")
        show = (
            self.transport.show_unit(
                machine=machine, unit_name=unit_name, timeout_s=STATUS_RPC_TIMEOUT_S
            )
            if unit_name
            else None
        )

        if state is None:
            state = {
                "status": "QUEUED",
                "started_at": None,
                "ended_at": None,
                "return_code": None,
                "retries_used": 0,
                "last_error": "",
            }

        if show is None:
            if (
                str(state.get("status") or "") == "RUNNING"
                and _heartbeat_age_seconds(state) > STALE_HEARTBEAT_SECONDS
            ):
                state["status"] = "UNKNOWN"
                state["last_error"] = "runner heartbeat is stale and systemd unit was not found"
            return state

        active_state = str(show.get("ActiveState") or "").lower()
        result = str(show.get("Result") or "").lower()
        exec_main_status = _int_or_none(show.get("ExecMainStatus"))
        if active_state in {"active", "activating", "reloading"}:
            state["status"] = "RUNNING" if state.get("started_at") else "QUEUED"
            return state
        if active_state == "failed":
            state["status"] = "FAILED"
            if state.get("ended_at") is None:
                state["ended_at"] = utc_now_iso()
            if state.get("return_code") is None and exec_main_status is not None:
                state["return_code"] = exec_main_status
            return state
        if active_state == "inactive" and state.get("status") == "RUNNING":
            if exec_main_status == 0 and result in {"success", ""}:
                state["status"] = "SUCCEEDED"
                state["return_code"] = 0
            elif result == "signal":
                state["status"] = "CANCELLED"
                state["return_code"] = exec_main_status or 143
            else:
                state["status"] = "FAILED"
                if state.get("return_code") is None:
                    state["return_code"] = exec_main_status if exec_main_status is not None else 1
            if state.get("ended_at") is None:
                state["ended_at"] = utc_now_iso()
        return state

    def _active_gpu_indices_by_node(self) -> dict[str, set[int]]:
        out: dict[str, set[int]] = self.db.active_gpu_leases_by_node()
        for row in self.db.list_active_jobs():
            placement = _placement_from_row(row)
            if not placement:
                continue
            for node in placement.get("nodes", []):
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("node_id") or "").strip()
                if not node_id:
                    continue
                visible = node.get("visible_gpu_indices") or []
                if isinstance(visible, list):
                    out.setdefault(node_id, set()).update(
                        int(idx) for idx in visible if _int_or_none(idx) is not None
                    )
            for worker in placement.get("workers", []):
                if not isinstance(worker, dict):
                    continue
                node_id = str(worker.get("node_id") or "").strip()
                physical_gpu = _int_or_none(worker.get("physical_gpu"))
                if node_id and physical_gpu is not None:
                    out.setdefault(node_id, set()).add(int(physical_gpu))
        return out

    def _start_handles_parallel(
        self,
        handles: list[dict[str, Any]],
        registry: ClusterRegistry,
        job_id: str,
        request: JobRequest,
    ) -> None:
        def _start(handle: dict[str, Any]) -> dict[str, Any]:
            machine = registry.get_machine(str(handle.get("host") or ""))
            if machine is None:
                raise RuntimeError(f"Unknown machine '{handle.get('host')}'")
            self.transport.start_unit(
                machine=machine,
                unit_name=str(handle["unit_name"]),
                config_path=str(handle["config_path"]),
                description=f"lab-orch {job_id} rank={handle.get('rank')}",
                timeout_s=START_RPC_TIMEOUT_S,
                properties=request.systemd_properties,
            )
            return {"unit_name": str(handle["unit_name"]), "host": machine.host}

        max_workers = min(max(1, len(handles)), 16)
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_start, handle): handle for handle in handles}
            for future in as_completed(future_map):
                handle = future_map[future]
                try:
                    event_payload = future.result()
                    self.db.insert_job_event(job_id, "unit_started", event_payload)
                except Exception as exc:
                    errors.append(f"{handle.get('unit_name')}: {exc}")
                    self.db.insert_job_event(
                        job_id,
                        "unit_start_failed",
                        {
                            "unit_name": str(handle.get("unit_name") or ""),
                            "host": str(handle.get("host") or ""),
                            "error": str(exc),
                        },
                    )

        if errors:
            raise RuntimeError("distributed unit start failed: " + " | ".join(errors[:3]))

    def _stop_handles_best_effort(
        self, handles: list[dict[str, Any]], registry: ClusterRegistry
    ) -> None:
        for handle in handles:
            machine = registry.get_machine(str(handle.get("host") or ""))
            if machine is None:
                continue
            try:
                self.transport.stop_unit(
                    machine=machine,
                    unit_name=str(handle.get("unit_name") or ""),
                    timeout_s=3.0,
                )
            except Exception:
                pass

    def _registry(self) -> ClusterRegistry:
        return load_cluster_registry(self.cluster_config)


def _machine_for_node(
    registry: ClusterRegistry, node: NodeSnapshot
) -> MachineSpec | None:
    for selector in (node.node_id, node.hostname, node.ip):
        if selector:
            machine = registry.get_machine(str(selector))
            if machine is not None:
                return machine
    return None


def _resolve_visible_gpu_indices(
    node: NodeSnapshot,
    request: JobRequest,
    reserved_gpu_indices: set[int],
) -> list[int]:
    raw_visible = request.env.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is not None and str(raw_visible).strip():
        requested = _parse_gpu_indices(str(raw_visible))
        _validate_gpu_indices(node=node, indices=requested, reserved=reserved_gpu_indices)
        return requested
    if request.gpus <= 0:
        return []
    gpu_count = _require_int_gpu_count(float(request.gpus))
    return _auto_select_gpu_indices(
        node=node, count=gpu_count, reserved_gpu_indices=reserved_gpu_indices
    )


def _auto_select_gpu_indices(
    node: NodeSnapshot, count: int, reserved_gpu_indices: set[int]
) -> list[int]:
    gpu_details = node.extras.get("gpu_details") if isinstance(node.extras, dict) else []
    if not isinstance(gpu_details, list):
        gpu_details = []
    candidates: list[tuple[float, float, int]] = []
    for item in gpu_details:
        if not isinstance(item, dict):
            continue
        index = _int_or_none(item.get("index"))
        if index is None or index in reserved_gpu_indices:
            continue
        proc_count = float(item.get("proc_count", 0.0) or 0.0)
        if proc_count > 0.0:
            continue
        util = float(item.get("util_percent", 0.0) or 0.0)
        mem_free = float(item.get("memory_free_mib", 0.0) or 0.0)
        candidates.append((util, -mem_free, int(index)))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen = [idx for _util, _neg_mem_free, idx in candidates[:count]]
    if len(chosen) < count:
        raise RuntimeError(
            f"Node '{node.hostname}' does not currently expose {count} free GPU indices."
        )
    return chosen


def _validate_gpu_indices(
    node: NodeSnapshot, indices: list[int], reserved: set[int]
) -> None:
    max_gpu = int(node.gpus_total)
    for idx in indices:
        if idx < 0 or idx >= max_gpu:
            raise RuntimeError(
                f"GPU index {idx} is out of range for node {node.hostname} (available: 0..{max(0, max_gpu - 1)})."
            )
        if idx in reserved:
            raise RuntimeError(
                f"GPU index {idx} on node {node.hostname} is already reserved by another active job."
            )


def _require_int_gpu_count(value: float) -> int:
    rounded = int(round(value))
    if abs(value - float(rounded)) > 1e-6:
        raise RuntimeError(f"GPU jobs require an integer GPU count. Got gpus={value}.")
    if rounded < 1:
        raise RuntimeError(f"GPU jobs require gpus >= 1. Got gpus={value}.")
    return rounded


def _write_runner_config(path: str, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_private(file_path, payload)


def _write_json_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _gpu_leases_for_node(
    job_id: str,
    node_id: str,
    node_hostname: str,
    gpu_indices: list[int],
) -> list[dict[str, Any]]:
    created_at = utc_now_iso()
    return [
        {
            "job_id": job_id,
            "node_id": node_id,
            "node_hostname": node_hostname,
            "gpu_index": int(idx),
            "created_at": created_at,
        }
        for idx in gpu_indices
    ]


def _gpu_leases_for_placement_nodes(
    job_id: str, placement_nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    leases: list[dict[str, Any]] = []
    for node in placement_nodes:
        visible = node.get("visible_gpu_indices") or []
        if not isinstance(visible, list):
            continue
        leases.extend(
            _gpu_leases_for_node(
                job_id=job_id,
                node_id=str(node.get("node_id") or ""),
                node_hostname=str(node.get("node_hostname") or ""),
                gpu_indices=[int(idx) for idx in visible],
            )
        )
    return leases


def _build_vgpu_manifest(
    handles: list[dict[str, Any]],
    visible_specs_by_unit: dict[str, str],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for handle in sorted(handles, key=lambda item: int(item.get("rank", 0))):
        unit_name = str(handle.get("unit_name") or "")
        manifest.append(
            {
                "vgpu_index": int(handle.get("rank", 0)),
                "host": str(handle.get("host") or ""),
                "node_id": str(handle.get("node_id") or ""),
                "node_hostname": str(handle.get("node_hostname") or ""),
                "node_ip": str(handle.get("node_ip") or ""),
                "node_rank": int(handle.get("node_rank", 0)),
                "local_rank": int(handle.get("local_rank", 0)),
                "physical_gpu": int(handle.get("physical_gpu", 0)),
                "cuda_visible_devices": str(visible_specs_by_unit.get(unit_name, "")),
            }
        )
    return manifest


def _inject_vgpu_envs_into_runner_configs(
    handles: list[dict[str, Any]],
    config_paths_by_unit: dict[str, str],
    visible_specs_by_unit: dict[str, str],
    vgpu_manifest: list[dict[str, Any]],
) -> None:
    manifest_json = json.dumps(vgpu_manifest, sort_keys=True)
    vgpu_ids = ",".join(str(entry["vgpu_index"]) for entry in vgpu_manifest)
    current_by_index = {
        int(entry["vgpu_index"]): entry for entry in vgpu_manifest if "vgpu_index" in entry
    }
    total = len(vgpu_manifest)
    for handle in handles:
        unit_name = str(handle.get("unit_name") or "")
        config_path = config_paths_by_unit.get(unit_name)
        if not config_path:
            continue
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        env = {str(k): str(v) for k, v in dict(payload.get("env") or {}).items()}
        rank = int(handle.get("rank", 0))
        current = current_by_index.get(rank, {})
        env.update(
            {
                "LAB_ORCH_VGPU": "1",
                "LAB_ORCH_VGPU_COUNT": str(total),
                "LAB_ORCH_VGPU_INDEX": str(rank),
                "LAB_ORCH_VGPU_IDS": vgpu_ids,
                "LAB_ORCH_VGPU_HOST": str(handle.get("node_hostname") or ""),
                "LAB_ORCH_VGPU_PHYSICAL_GPU": str(int(handle.get("physical_gpu", 0))),
                "LAB_ORCH_VGPU_VISIBLE_DEVICES": str(
                    visible_specs_by_unit.get(unit_name, "")
                ),
                "LAB_ORCH_VGPU_MANIFEST_JSON": manifest_json,
                "LAB_ORCH_VGPU_CURRENT_JSON": json.dumps(current, sort_keys=True),
            }
        )
        payload["env"] = env
        _write_runner_config(config_path, payload)


def _write_runner_state_placeholder(path: str, job_id: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "status": "QUEUED",
        "pid": None,
        "return_code": None,
        "started_at": None,
        "ended_at": None,
        "attempt": 0,
        "max_retries": 0,
        "retries_used": 0,
        "last_error": "",
        "last_heartbeat_at": None,
    }
    _write_json_private(file_path, payload)


def _load_runner_state(path: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _remote_handles_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("remote_handles_json")
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return [dict(item) for item in payload if isinstance(item, dict)]
        except Exception:
            pass
    unit_name = row.get("remote_unit_name")
    if unit_name:
        return [
            {
                "host": row.get("node_id") or row.get("node_hostname"),
                "unit_name": unit_name,
                "state_path": row.get("state_path"),
                "log_path": row.get("log_path"),
                "node_id": row.get("node_id"),
                "node_ip": row.get("node_ip"),
                "node_hostname": row.get("node_hostname"),
            }
        ]
    return []


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
    prev_status: str, states: list[dict[str, Any]]
) -> tuple[str, str | None, str | None, int | None]:
    if not states:
        return "UNKNOWN", None, None, None

    statuses = [str(state.get("status") or "UNKNOWN") for state in states]
    started_candidates = [
        str(state.get("started_at"))
        for state in states
        if state.get("started_at") not in {None, ""}
    ]
    ended_candidates = [
        str(state.get("ended_at"))
        for state in states
        if state.get("ended_at") not in {None, ""}
    ]
    started_at = min(started_candidates) if started_candidates else None

    if prev_status == "CANCELLED":
        status = "CANCELLED"
    elif "FAILED" in statuses:
        status = "FAILED"
    elif "CANCELLED" in statuses:
        status = "CANCELLED"
    elif "RUNNING" in statuses:
        status = "RUNNING"
    elif all(item == "SUCCEEDED" for item in statuses):
        status = "SUCCEEDED"
    elif all(item == "QUEUED" for item in statuses):
        status = "QUEUED"
    else:
        status = "RUNNING"

    all_terminal = all(item in TERMINAL_STATUSES for item in statuses)
    ended_at = max(ended_candidates) if ended_candidates and all_terminal else None

    codes = [
        _int_or_none(state.get("return_code"))
        for state in states
        if _int_or_none(state.get("return_code")) is not None
    ]
    return_code: int | None = None
    if status == "SUCCEEDED":
        return_code = 0
    elif status in {"FAILED", "CANCELLED"} and codes:
        return_code = next((code for code in codes if code not in {None, 0}), codes[0])

    return status, started_at, ended_at, return_code


def _max_retries_used(states: list[dict[str, Any]]) -> int:
    max_used = 0
    for state in states:
        raw = _int_or_none(state.get("retries_used"))
        if raw is not None:
            max_used = max(max_used, int(raw))
    return max_used


def _heartbeat_age_seconds(state: dict[str, Any]) -> float:
    raw = state.get("last_heartbeat_at") or state.get("started_at")
    if raw in {None, ""}:
        return float("inf")
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(text)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(tz=timezone.utc) - timestamp).total_seconds())
    except Exception:
        return float("inf")


def _parse_gpu_indices(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        idx = int(token)
        if idx in seen:
            raise RuntimeError(f"duplicate GPU index {idx} in CUDA_VISIBLE_DEVICES")
        seen.add(idx)
        indices.append(idx)
    return indices


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _read_log_file(path: Path, n_lines: int | None) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if n_lines is None:
            return handle.read()
        lines = handle.readlines()
    return "".join(lines[-max(1, int(n_lines)) :])


def _log_sources_for_row(row: dict[str, Any], logs_dir: Path) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    placement = _placement_from_row(row)
    job_id = str(row["job_id"])
    handles = _remote_handles_from_row(row)

    if (row.get("job_mode") or "single") == "single":
        handle = handles[0] if handles else {}
        stdout_path = str(
            handle.get("stdout_path") or handle.get("log_path") or row.get("log_path") or (logs_dir / f"{job_id}.log")
        )
        stderr_path = str(
            handle.get("stderr_path") or _default_stderr_log_path(stdout_path)
        )
        sources.append(
            {"label": f"job={job_id} stdout", "path": stdout_path, "stream": "stdout"}
        )
        sources.append(
            {"label": f"job={job_id} stderr", "path": stderr_path, "stream": "stderr"}
        )
        return sources

    workers = placement.get("workers", []) if isinstance(placement, dict) else []
    for idx, handle in enumerate(handles):
        worker = workers[idx] if idx < len(workers) and isinstance(workers[idx], dict) else {}
        host = worker.get("node_hostname") or handle.get("node_hostname")
        base_label = f"rank={handle.get('rank', idx)} unit={handle.get('unit_name')}"
        if host:
            base_label += f" host={host}"
        stdout_path = str(
            handle.get("stdout_path")
            or handle.get("log_path")
            or worker.get("stdout_path")
            or worker.get("log_path")
            or (logs_dir / f"{job_id}.rank{idx}.log")
        )
        stderr_path = str(
            handle.get("stderr_path")
            or worker.get("stderr_path")
            or _default_stderr_log_path(stdout_path)
        )
        sources.append(
            {"label": f"{base_label} stdout", "path": stdout_path, "stream": "stdout"}
        )
        sources.append(
            {"label": f"{base_label} stderr", "path": stderr_path, "stream": "stderr"}
        )

    if not sources:
        fallback = str(row.get("log_path") or (logs_dir / f"{job_id}.distributed.log"))
        sources.append({"label": f"job={job_id} stdout", "path": fallback, "stream": "stdout"})
    return sources


def _default_stderr_log_path(stdout_path: str) -> str:
    path = Path(stdout_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}.stderr{path.suffix}"))
    return str(path.with_name(path.name + ".stderr"))
