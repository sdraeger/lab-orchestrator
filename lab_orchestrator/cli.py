from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .app_config import AppConfig, default_config_file, init_app_config, load_app_config, save_app_config
from .models import GpuBinding, JobRequest
from .orchestrator import Orchestrator
from .policy import StaticSubmissionPolicy, load_submission_policy
from .registry import MachineSpec, load_cluster_registry
from .scheduler import format_capacity_brief, resolve_placement_strategy
from .utils import parse_env_pairs

SUBCOMMANDS = {
    "init-config",
    "register-machine",
    "unregister-machine",
    "machines",
    "overview",
    "doctor",
    "submit",
    "run",
    "jobs",
    "status",
    "cancel",
    "logs",
}
GLOBAL_OPTIONS_WITH_VALUES = {
    "--config-file",
    "--cluster-config",
    "--db",
    "--logs-dir",
    "--state-dir",
    "--scheduler",
    "--policy-config",
    "--max-active-jobs-per-user",
    "--max-cpus-per-user",
    "--max-gpus-per-user",
    "--allow-host",
    "--deny-host",
}


def main() -> None:
    raw_argv = sys.argv[1:]
    parser = build_implicit_run_parser() if _wants_implicit_run(raw_argv) else build_parser()
    args = parser.parse_args(raw_argv)
    apply_runtime_defaults(args)

    if not getattr(args, "cmd", None):
        parser.print_help()
        return

    try:
        if args.cmd == "init-config":
            config = init_app_config(args.root)
            save_app_config(args.config_file, config)
            print_init_config(config=config, config_file=args.config_file)
            return

        if args.cmd == "register-machine":
            orch = _build_orchestrator(args)
            machine = MachineSpec(
                host=str(args.host),
                ssh_host=str(args.ssh_host) if args.ssh_host else None,
                ssh_user=str(args.ssh_user) if args.ssh_user else None,
                python_bin=str(args.python_bin or "python3"),
                local=bool(args.local),
                labels=[str(item).strip() for item in (args.label or []) if str(item).strip()],
            )
            orch.register_machine(machine)
            print_machine(machine, action="registered", cluster_config=args.cluster_config)
            return

        if args.cmd == "unregister-machine":
            orch = _build_orchestrator(args)
            removed = orch.unregister_machine(args.host)
            print_machine(
                removed, action="unregistered", cluster_config=args.cluster_config
            )
            return

        if args.cmd == "machines":
            registry = load_cluster_registry(args.cluster_config)
            print_machines(registry.machines)
            return

        policy = build_submission_policy(args)
        strategy = resolve_placement_strategy(args.scheduler)
        orch = Orchestrator(
            cluster_config=args.cluster_config,
            db_path=args.db,
            logs_dir=args.logs_dir,
            state_dir=args.state_dir,
            placement_strategy=strategy,
            submission_policy=policy,
        )

        if args.cmd == "overview":
            payload = orch.overview()
            print_overview(
                payload["nodes"],
                payload["reservations"],
                payload.get("gpu_index_reservations") or {},
            )
            return

        if args.cmd == "doctor":
            rows = orch.doctor()
            print_doctor(rows)
            return

        if args.cmd == "submit":
            request, target_hosts = build_submit_request(args)
            apply_target_host_override(policy, target_hosts)
            row = orch.submit(request)
            print_submit_result(
                row=row,
                scheduler_name=args.scheduler,
                policy_name=policy.name,
                max_retries=request.max_retries,
            )
            return

        if args.cmd == "run":
            request, target_hosts = build_run_request(args)
            apply_target_host_override(policy, target_hosts)
            row = orch.submit(request)
            print_submit_result(
                row=row,
                scheduler_name=args.scheduler,
                policy_name=policy.name,
                max_retries=request.max_retries,
            )
            if args.detach:
                return
            follow_logs(
                orch=orch,
                job_id=str(row["job_id"]),
                n_lines=args.lines,
                poll_seconds=args.poll_seconds,
            )
            final = orch.status(str(row["job_id"]), refresh=True)
            status = str(final.get("status") or "")
            if status != "SUCCEEDED":
                raw_return_code = final.get("return_code")
                return_code = int(raw_return_code) if raw_return_code is not None else 1
                if return_code == 0:
                    return_code = 1
                print(
                    f"run failed: job_id={row['job_id']} status={status} return_code={return_code}",
                    file=sys.stderr,
                )
                sys.exit(return_code)
            print(f"completed job_id={row['job_id']} status={status}")
            return

        if args.cmd == "jobs":
            rows = orch.list_jobs(limit=args.limit, refresh=args.refresh)
            print_jobs(rows)
            return

        if args.cmd == "status":
            row = orch.status(args.job_id, refresh=(not args.no_refresh))
            print_job_status(row)
            return

        if args.cmd == "cancel":
            row = orch.cancel(args.job_id, grace_seconds=args.grace_seconds)
            print(
                f"job_id={row['job_id']} status={row['status']} return_code={row['return_code']}"
            )
            return

        if args.cmd == "logs":
            if args.follow:
                follow_logs(
                    orch=orch,
                    job_id=args.job_id,
                    n_lines=args.lines,
                    poll_seconds=args.poll_seconds,
                )
                return
            row = orch.status(args.job_id, refresh=True)
            if args.all or str(row.get("status")) == "FAILED":
                text = orch.logs_all(args.job_id)
            else:
                text = orch.logs(args.job_id, n_lines=args.lines)
            print(text, end="")
            return

        raise RuntimeError(f"unknown command {args.cmd}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        description="SSH/systemd-based job orchestration for lab machines",
    )
    parser.add_argument(
        "--config-file",
        default=os.environ.get("LAB_ORCH_CONFIG", str(default_config_file())),
        help="Path to app config that defines default cluster/db/log/state locations",
    )
    parser.add_argument(
        "--cluster-config",
        default=None,
        help="YAML file containing registered machines",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory for job logs",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for runner state and spool files",
    )
    parser.add_argument(
        "--scheduler",
        default=os.environ.get("LAB_ORCH_SCHEDULER", "balanced"),
        help="Placement strategy: balanced or fair-share",
    )
    parser.add_argument(
        "--policy-config",
        default=os.environ.get("LAB_ORCH_POLICY_CONFIG"),
        help="Optional YAML policy config for quotas and host filters",
    )
    parser.add_argument(
        "--max-active-jobs-per-user",
        type=int,
        help="Override policy limit: max active jobs per submitter",
    )
    parser.add_argument(
        "--max-cpus-per-user",
        type=float,
        help="Override policy limit: max active CPUs per submitter",
    )
    parser.add_argument(
        "--max-gpus-per-user",
        type=float,
        help="Override policy limit: max active GPUs per submitter",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Allowlist host/ip/node_id for scheduling (repeatable)",
    )
    parser.add_argument(
        "--deny-host",
        action="append",
        default=[],
        help="Denylist host/ip/node_id for scheduling (repeatable)",
    )

    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser(
        "init-config",
        help="Create a persistent app config pointing at a shared lab-orch root",
    )
    p_init.add_argument(
        "--root",
        required=True,
        help="Shared root directory where cluster.yaml, jobs.db, logs/, and state/ will live",
    )

    p_register = sub.add_parser(
        "register-machine", help="Register or update a machine in the cluster config"
    )
    p_register.add_argument("--host", required=True, help="Stable machine name")
    p_register.add_argument(
        "--ssh-host", help="SSH hostname/address if it differs from --host"
    )
    p_register.add_argument("--ssh-user", help="SSH username override")
    p_register.add_argument(
        "--python-bin",
        default="python3",
        help="Python binary on the target host (default: python3)",
    )
    p_register.add_argument(
        "--local",
        action="store_true",
        help="Treat this machine as local and execute without SSH",
    )
    p_register.add_argument(
        "--label", action="append", default=[], help="Optional machine label"
    )

    p_unregister = sub.add_parser(
        "unregister-machine", help="Remove a machine from the cluster config"
    )
    p_unregister.add_argument("--host", required=True)

    sub.add_parser("machines", help="List registered machines")
    sub.add_parser("overview", help="Show cluster-wide node resources and load")
    sub.add_parser("doctor", help="Check SSH, Python, systemd, and probe health")

    p_submit = sub.add_parser(
        "submit",
        help="Schedule a script on one node or as a distributed multi-node job",
    )
    p_submit.add_argument(
        "--job-config", help="YAML file with name/command/cpus/gpus/workdir/env"
    )
    p_submit.add_argument("--name", help="Job name")
    p_submit.add_argument("--command", help="Shell command to run")
    p_submit.add_argument("--cpus", type=float, help="Requested CPUs")
    p_submit.add_argument("--gpus", type=float, help="Requested GPUs")
    p_submit.add_argument(
        "--target-host",
        help="Pin scheduling to one host/ip/node_id (must be a registered machine)",
    )
    p_submit.add_argument(
        "--visible-gpus",
        help="Comma-separated GPU indices to expose via CUDA_VISIBLE_DEVICES (requires --target-host)",
    )
    p_submit.add_argument(
        "--gpu-bind",
        action="append",
        default=[],
        help=(
            "Explicit host/GPU bindings in HOST:GPU[,GPU...] form; repeatable "
            "(for example: --gpu-bind node1:0,2 --gpu-bind node2:1)"
        ),
    )
    p_submit.add_argument("--workdir", help="Command working directory")
    p_submit.add_argument(
        "--submit-user", help="Submission owner for fair-share/policy"
    )
    p_submit.add_argument(
        "--distributed",
        action="store_true",
        help="Force distributed mode (one process per GPU across nodes)",
    )
    p_submit.add_argument(
        "--max-retries",
        type=int,
        help="Retry command this many times on non-zero exit",
    )
    p_submit.add_argument(
        "--retry-backoff-seconds",
        type=float,
        help="Delay between command retries",
    )
    p_submit.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE format (repeatable)",
    )
    p_submit.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Custom metadata in KEY=VALUE format (repeatable)",
    )
    add_systemd_property_args(p_submit)

    p_run = sub.add_parser(
        "run",
        help="Run any shell command immediately, optionally pinned to a host or GPU set",
    )
    p_run.add_argument(
        "--target-host",
        "--host",
        dest="target_host",
        help="Host/ip/node_id where the command should run",
    )
    p_run.add_argument(
        "--gpus",
        type=float,
        help="Requested GPUs for scheduling or host-pinned execution",
    )
    p_run.add_argument(
        "--visible-gpus",
        help="Comma-separated GPU indices to expose inside the process (for example: 0,2)",
    )
    p_run.add_argument(
        "--gpu-bind",
        action="append",
        default=[],
        help=(
            "Explicit host/GPU bindings in HOST:GPU[,GPU...] form; repeatable "
            "(for example: --gpu-bind node1:0,2 --gpu-bind node2:1)"
        ),
    )
    p_run.add_argument(
        "--distributed",
        action="store_true",
        help="Force distributed placement when multiple GPUs are requested",
    )
    p_run.add_argument("--name", help="Optional job name")
    p_run.add_argument(
        "--cpus",
        type=float,
        default=1.0,
        help="CPU reservation for scheduling (default: 1.0)",
    )
    p_run.add_argument(
        "--workdir",
        help="Command working directory (default: current directory)",
    )
    p_run.add_argument(
        "--submit-user",
        help="Submission owner for fair-share/policy",
    )
    p_run.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry command this many times on non-zero exit",
    )
    p_run.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Delay between command retries",
    )
    p_run.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE format (repeatable)",
    )
    p_run.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Custom metadata in KEY=VALUE format (repeatable)",
    )
    add_systemd_property_args(p_run)
    p_run.add_argument(
        "--detach",
        action="store_true",
        help="Submit and return immediately instead of streaming logs until completion",
    )
    p_run.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Initial tail lines when streaming logs (ignored with --detach)",
    )
    p_run.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval when streaming logs (ignored with --detach)",
    )
    p_run.add_argument(
        "cmd_parts",
        nargs=argparse.REMAINDER,
        help="Command to run. Use '--' before the command to pass through args cleanly.",
    )

    p_jobs = sub.add_parser("jobs", help="List jobs")
    p_jobs.add_argument("--limit", type=int, default=50)
    p_jobs.add_argument(
        "--refresh", action="store_true", help="Refresh running status from machines"
    )

    p_status = sub.add_parser("status", help="Show detailed job status")
    p_status.add_argument("job_id")
    p_status.add_argument("--no-refresh", action="store_true")

    p_cancel = sub.add_parser("cancel", help="Cancel a running job")
    p_cancel.add_argument("job_id")
    p_cancel.add_argument("--grace-seconds", type=int, default=20)

    p_logs = sub.add_parser("logs", help="Show job logs (tail/full/follow)")
    p_logs.add_argument("job_id")
    p_logs.add_argument("--lines", type=int, default=100)
    p_logs.add_argument(
        "-f", "--follow", action="store_true", help="Stream logs while job is alive"
    )
    p_logs.add_argument(
        "--all",
        action="store_true",
        help="Print full log history (failed jobs already default to full)",
    )
    p_logs.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for --follow",
    )
    return parser


def build_implicit_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        description="Run a command on the lab cluster with minimal wrapping",
    )
    parser.add_argument(
        "--config-file",
        default=os.environ.get("LAB_ORCH_CONFIG", str(default_config_file())),
        help="Path to app config that defines default cluster/db/log/state locations",
    )
    parser.add_argument(
        "--cluster-config",
        default=None,
        help="YAML file containing registered machines",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite DB path",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory for job logs",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for runner state and spool files",
    )
    parser.add_argument(
        "--scheduler",
        default=os.environ.get("LAB_ORCH_SCHEDULER", "balanced"),
        help="Placement strategy: balanced or fair-share",
    )
    parser.add_argument(
        "--policy-config",
        default=os.environ.get("LAB_ORCH_POLICY_CONFIG"),
        help="Optional YAML policy config for quotas and host filters",
    )
    parser.add_argument(
        "--max-active-jobs-per-user",
        type=int,
        help="Override policy limit: max active jobs per submitter",
    )
    parser.add_argument(
        "--max-cpus-per-user",
        type=float,
        help="Override policy limit: max active CPUs per submitter",
    )
    parser.add_argument(
        "--max-gpus-per-user",
        type=float,
        help="Override policy limit: max active GPUs per submitter",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Allowlist host/ip/node_id for scheduling (repeatable)",
    )
    parser.add_argument(
        "--deny-host",
        action="append",
        default=[],
        help="Denylist host/ip/node_id for scheduling (repeatable)",
    )
    parser.add_argument(
        "--target-host",
        "--host",
        dest="target_host",
        help="Host/ip/node_id where the command should run",
    )
    parser.add_argument(
        "--gpus",
        type=float,
        help="Requested GPUs for scheduling or host-pinned execution",
    )
    parser.add_argument(
        "--visible-gpus",
        help="Comma-separated GPU indices to expose inside the process (for example: 0,2)",
    )
    parser.add_argument(
        "--gpu-bind",
        action="append",
        default=[],
        help=(
            "Explicit host/GPU bindings in HOST:GPU[,GPU...] form; repeatable "
            "(for example: --gpu-bind node1:0,2 --gpu-bind node2:1)"
        ),
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Force distributed placement when multiple GPUs are requested",
    )
    parser.add_argument("--name", help="Optional job name")
    parser.add_argument(
        "--cpus",
        type=float,
        default=1.0,
        help="CPU reservation for scheduling (default: 1.0)",
    )
    parser.add_argument(
        "--workdir",
        help="Command working directory (default: current directory)",
    )
    parser.add_argument(
        "--submit-user",
        help="Submission owner for fair-share/policy",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry command this many times on non-zero exit",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Delay between command retries",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE format (repeatable)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Custom metadata in KEY=VALUE format (repeatable)",
    )
    add_systemd_property_args(parser)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Submit and return immediately instead of streaming logs until completion",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Initial tail lines when streaming logs (ignored with --detach)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval when streaming logs (ignored with --detach)",
    )
    parser.add_argument(
        "cmd_parts",
        nargs=argparse.REMAINDER,
        help="Command to run directly, for example: orch --gpus 1 python train.py",
    )
    parser.set_defaults(cmd="run")
    return parser


def build_submit_request(args: argparse.Namespace) -> tuple[JobRequest, list[str]]:
    cfg = load_job_config(args.job_config) if args.job_config else {}
    env_from_cfg = cfg.get("env", {})
    if env_from_cfg is None:
        env_from_cfg = {}
    if not isinstance(env_from_cfg, dict):
        raise ValueError("submit config field 'env' must be a mapping of KEY: VALUE")

    metadata_from_cfg = cfg.get("metadata", {})
    if metadata_from_cfg is None:
        metadata_from_cfg = {}
    if not isinstance(metadata_from_cfg, dict):
        raise ValueError(
            "submit config field 'metadata' must be a mapping of KEY: VALUE"
        )

    name = args.name or cfg.get("name")
    command = args.command or cfg.get("command")
    cpus = args.cpus if args.cpus is not None else float(cfg.get("cpus", 1.0))
    if args.gpus is not None:
        gpus = float(args.gpus)
        gpus_source = "cli"
    elif cfg.get("gpus") is not None:
        gpus = float(cfg.get("gpus"))
        gpus_source = "config"
    else:
        gpus = 0.0
        gpus_source = "default"
    workdir = args.workdir or cfg.get("workdir") or os.getcwd()
    distributed_cfg = bool(cfg.get("distributed", False))
    distributed = bool(args.distributed or distributed_cfg)
    submit_user = args.submit_user or cfg.get("submit_user")
    max_retries = (
        int(args.max_retries)
        if args.max_retries is not None
        else int(cfg.get("max_retries", 0))
    )
    retry_backoff_seconds = (
        float(args.retry_backoff_seconds)
        if args.retry_backoff_seconds is not None
        else float(cfg.get("retry_backoff_seconds", 5.0))
    )
    target_host = args.target_host or cfg.get("target_host")
    visible_spec = (
        args.visible_gpus if args.visible_gpus is not None else cfg.get("visible_gpus")
    )
    visible_gpus = parse_visible_gpu_selection(visible_spec)
    gpu_bind_raw = list(args.gpu_bind or [])
    if cfg.get("gpu_bind") is not None and not gpu_bind_raw:
        gpu_bind_raw = (
            list(cfg.get("gpu_bind"))
            if isinstance(cfg.get("gpu_bind"), list)
            else [cfg.get("gpu_bind")]
        )
    explicit_gpu_bindings = parse_gpu_bindings(gpu_bind_raw)

    if not name:
        raise ValueError("submit requires --name (or a 'name' in --job-config)")
    if not command:
        raise ValueError("submit requires --command (or a 'command' in --job-config)")
    if max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be >= 0")

    if explicit_gpu_bindings:
        if visible_gpus or target_host:
            raise ValueError(
                "Use either --gpu-bind or --target-host with --visible-gpus, not both"
            )
        expected_gpus = float(len(explicit_gpu_bindings))
        if gpus_source != "default" and abs(gpus - expected_gpus) > 1e-9:
            raise ValueError(
                "--gpus must match the number of explicit --gpu-bind entries "
                f"(expected {int(expected_gpus)}, got {gpus:g})"
            )
        gpus = expected_gpus
        distributed = True
    else:
        if visible_gpus and not target_host:
            raise ValueError("--visible-gpus requires --target-host")
        if visible_gpus and distributed:
            raise ValueError(
                "--visible-gpus currently supports single-node submissions"
            )
        if visible_gpus:
            expected_gpus = float(len(visible_gpus))
            if gpus_source != "default" and abs(gpus - expected_gpus) > 1e-9:
                raise ValueError(
                    "--gpus must match the number of --visible-gpus entries "
                    f"(expected {int(expected_gpus)}, got {gpus:g})"
                )
            gpus = expected_gpus

    merged_env = {str(k): str(v) for k, v in env_from_cfg.items()}
    merged_env.update(parse_env_pairs(args.env))
    if visible_gpus and not explicit_gpu_bindings:
        merged_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(idx) for idx in visible_gpus)

    merged_metadata = {str(k): str(v) for k, v in metadata_from_cfg.items()}
    merged_metadata.update(parse_env_pairs(args.metadata))
    if target_host and not explicit_gpu_bindings:
        merged_metadata.setdefault("target_host", str(target_host))
    if visible_gpus and not explicit_gpu_bindings:
        merged_metadata.setdefault(
            "visible_gpus", ",".join(str(idx) for idx in visible_gpus)
        )
    if explicit_gpu_bindings:
        merged_metadata.setdefault(
            "explicit_gpu_bindings",
            ",".join(f"{b.host}:{b.gpu_index}" for b in explicit_gpu_bindings),
        )

    if explicit_gpu_bindings:
        target_hosts = sorted({b.host for b in explicit_gpu_bindings})
    elif target_host:
        target_hosts = [str(target_host)]
    else:
        target_hosts = []

    return (
        JobRequest(
            name=str(name),
            command=str(command),
            command_argv=[],
            use_shell=True,
            cpus=float(cpus),
            gpus=float(gpus),
            workdir=str(workdir),
            env=merged_env,
            distributed=distributed,
            submit_user=str(submit_user) if submit_user else None,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            metadata=merged_metadata,
            explicit_gpu_bindings=explicit_gpu_bindings,
            systemd_properties=build_systemd_properties(args),
        ),
        target_hosts,
    )


def build_run_request(args: argparse.Namespace) -> tuple[JobRequest, list[str]]:
    max_retries = int(args.max_retries)
    retry_backoff_seconds = float(args.retry_backoff_seconds)
    if max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if retry_backoff_seconds < 0:
        raise ValueError("--retry-backoff-seconds must be >= 0")

    cmd_parts = list(args.cmd_parts)
    if cmd_parts and cmd_parts[0] == "--":
        cmd_parts = cmd_parts[1:]
    if not cmd_parts:
        raise ValueError("run requires a command after '--'")
    command = " ".join(shlex.quote(part) for part in cmd_parts)

    explicit_gpu_bindings = parse_gpu_bindings(args.gpu_bind)
    visible_gpus = parse_visible_gpu_selection(args.visible_gpus)
    has_target = bool(args.target_host)

    merged_env = parse_env_pairs(args.env)
    merged_metadata = parse_env_pairs(args.metadata)
    merged_metadata.setdefault("run_mode", "shell")

    if explicit_gpu_bindings:
        if has_target or visible_gpus or args.gpus is not None:
            raise ValueError(
                "Use either --gpu-bind or host/GPU request flags, not both"
            )
        gpus = float(len(explicit_gpu_bindings))
        target_hosts = sorted({b.host for b in explicit_gpu_bindings})
        merged_metadata.setdefault(
            "explicit_gpu_bindings",
            ",".join(f"{b.host}:{b.gpu_index}" for b in explicit_gpu_bindings),
        )
    else:
        if visible_gpus and not has_target:
            raise ValueError("--visible-gpus requires --target-host/--host")
        requested_gpus = float(args.gpus) if args.gpus is not None else None
        if requested_gpus is not None and requested_gpus < 0:
            raise ValueError("--gpus must be >= 0")
        if visible_gpus:
            merged_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(idx) for idx in visible_gpus
            )
            visible_gpu_count = float(len(visible_gpus))
            if requested_gpus is not None and abs(requested_gpus - visible_gpu_count) > 1e-9:
                raise ValueError(
                    "--gpus must match the number of --visible-gpus entries "
                    f"(expected {int(visible_gpu_count)}, got {requested_gpus:g})"
                )
            gpus = visible_gpu_count
            merged_metadata.setdefault(
                "visible_gpus", ",".join(str(idx) for idx in visible_gpus)
            )
        else:
            gpus = requested_gpus if requested_gpus is not None else 0.0
        if has_target:
            merged_metadata.setdefault("target_host", str(args.target_host))
            target_hosts = [str(args.target_host)]
        else:
            target_hosts = []

    default_name = f"run-{Path(cmd_parts[0]).name or 'command'}"
    return (
        JobRequest(
            name=str(args.name or default_name),
            command=command,
            command_argv=[str(part) for part in cmd_parts],
            use_shell=False,
            cpus=float(args.cpus),
            gpus=gpus,
            workdir=str(args.workdir or os.getcwd()),
            env=merged_env,
            distributed=bool(explicit_gpu_bindings or getattr(args, "distributed", False)),
            submit_user=str(args.submit_user) if args.submit_user else None,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            metadata=merged_metadata,
            explicit_gpu_bindings=explicit_gpu_bindings,
            systemd_properties=build_systemd_properties(args),
        ),
        target_hosts,
    )


def add_systemd_property_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-max-seconds",
        type=float,
        help="Optional systemd RuntimeMaxSec limit for the remote unit",
    )
    parser.add_argument(
        "--memory-max",
        help="Optional systemd MemoryMax value for the remote unit, e.g. 64G",
    )
    parser.add_argument(
        "--tasks-max",
        type=int,
        help="Optional systemd TasksMax value for the remote unit",
    )
    parser.add_argument(
        "--cpu-quota-percent",
        type=float,
        help="Optional systemd CPUQuota percentage for the remote unit",
    )


def build_systemd_properties(args: argparse.Namespace) -> dict[str, str]:
    properties: dict[str, str] = {}
    runtime = getattr(args, "runtime_max_seconds", None)
    if runtime is not None:
        runtime_value = float(runtime)
        if runtime_value <= 0:
            raise ValueError("--runtime-max-seconds must be > 0")
        properties["RuntimeMaxSec"] = f"{runtime_value:g}s"
    memory_max = getattr(args, "memory_max", None)
    if memory_max:
        properties["MemoryMax"] = str(memory_max)
    tasks_max = getattr(args, "tasks_max", None)
    if tasks_max is not None:
        if int(tasks_max) <= 0:
            raise ValueError("--tasks-max must be > 0")
        properties["TasksMax"] = str(int(tasks_max))
    cpu_quota = getattr(args, "cpu_quota_percent", None)
    if cpu_quota is not None:
        quota_value = float(cpu_quota)
        if quota_value <= 0:
            raise ValueError("--cpu-quota-percent must be > 0")
        properties["CPUQuota"] = f"{quota_value:g}%"
    return properties


def parse_gpu_bindings(value: Any) -> list[GpuBinding]:
    if value is None:
        return []
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = [str(v) for v in value if str(v).strip()]
    else:
        raise ValueError("--gpu-bind expects HOST:GPU[,GPU...] strings (repeatable)")

    bindings: list[GpuBinding] = []
    seen: set[tuple[str, int]] = set()
    for item in raw_items:
        for entry in [s.strip() for s in str(item).split(";") if s.strip()]:
            if ":" not in entry:
                raise ValueError(
                    f"--gpu-bind entry '{entry}' must be in HOST:GPU[,GPU...] format"
                )
            host_raw, gpu_raw = entry.split(":", 1)
            host = host_raw.strip()
            if not host:
                raise ValueError("--gpu-bind host cannot be empty")
            gpu_indices = parse_visible_gpu_selection(gpu_raw)
            if not gpu_indices:
                raise ValueError(f"--gpu-bind entry '{entry}' has no GPU indices")
            for gpu_idx in gpu_indices:
                key = (host.lower(), int(gpu_idx))
                if key in seen:
                    raise ValueError(
                        f"--gpu-bind has duplicate host/GPU pair {host}:{gpu_idx}"
                    )
                seen.add(key)
                bindings.append(GpuBinding(host=host, gpu_index=int(gpu_idx)))
    return bindings


def parse_visible_gpu_selection(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        flat = ",".join(str(v) for v in value)
        return _parse_visible_gpu_indices(flat)
    if isinstance(value, str):
        return _parse_visible_gpu_indices(value)
    raise ValueError(
        "visible_gpus must be a comma-separated string or list of integer indices"
    )


def _parse_visible_gpu_indices(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise ValueError("--visible-gpus contains an empty item")
        try:
            idx = int(token)
        except Exception as exc:
            raise ValueError(f"--visible-gpus expects integers, got '{token}'") from exc
        if idx < 0:
            raise ValueError("--visible-gpus indices must be >= 0")
        if idx in seen:
            raise ValueError(f"--visible-gpus has duplicate index {idx}")
        seen.add(idx)
        indices.append(idx)
    return indices


def apply_target_host_override(
    policy: StaticSubmissionPolicy,
    target_host: str | list[str] | tuple[str, ...] | None,
) -> None:
    if not target_host:
        return
    items: list[str]
    if isinstance(target_host, str):
        items = [target_host]
    else:
        items = [str(v) for v in target_host]

    normalized_hosts: set[str] = set()
    for raw in items:
        normalized = str(raw).strip().lower()
        if not normalized:
            raise ValueError("--target-host cannot be empty")
        if normalized in policy.denied_hosts:
            raise ValueError(
                f"target host '{raw}' is denied by active policy configuration"
            )
        if policy.allowed_hosts and normalized not in policy.allowed_hosts:
            raise ValueError(
                f"target host '{raw}' is not present in the active allowlist"
            )
        normalized_hosts.add(normalized)
    if normalized_hosts:
        policy.allowed_hosts = normalized_hosts


def _prog_name() -> str:
    argv0 = Path(sys.argv[0]).name.strip()
    return argv0 or "lab-orch"


def _wants_implicit_run(argv: list[str]) -> bool:
    if not argv:
        return False
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            return True
        if token in {"-h", "--help"}:
            return False
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            if option in GLOBAL_OPTIONS_WITH_VALUES and "=" not in token:
                i += 2
                continue
            if option in GLOBAL_OPTIONS_WITH_VALUES:
                i += 1
                continue
            return True
        if token.startswith("-") and token != "-":
            return True
        return token not in SUBCOMMANDS
    return False


def print_machine(machine: MachineSpec, action: str, cluster_config: str) -> None:
    mode = "local" if machine.local else "ssh"
    target = machine.ssh_host or machine.host
    user = machine.ssh_user or "-"
    print(
        f"{action} machine host={machine.host} target={target} mode={mode} ssh_user={user} config={cluster_config}"
    )


def print_machines(machines: list[MachineSpec]) -> None:
    if not machines:
        print("No registered machines.")
        return
    header = f"{'HOST':<20} {'TARGET':<20} {'MODE':<8} {'SSH USER':<12} {'PYTHON':<12} LABELS"
    print(header)
    print("-" * len(header))
    for machine in machines:
        print(
            f"{machine.host[:20]:<20} {machine.connect_host[:20]:<20} "
            f"{('local' if machine.local else 'ssh'):<8} "
            f"{(machine.ssh_user or '-')[:12]:<12} {machine.python_bin[:12]:<12} "
            f"{','.join(machine.labels)}"
        )


def print_init_config(config: AppConfig, config_file: str) -> None:
    print(f"wrote config: {config_file}")
    print(f"cluster_config: {config.cluster_config}")
    print(f"db: {config.db}")
    print(f"logs_dir: {config.logs_dir}")
    print(f"state_dir: {config.state_dir}")
    print()
    print("You can now run `lab-orch` without exporting LAB_ORCH_* variables.")


def print_submit_result(
    row: dict[str, Any], scheduler_name: str, policy_name: str, max_retries: int
) -> None:
    print(
        f"submitted job_id={row['job_id']} status={row['status']} "
        f"mode={row.get('job_mode', 'single')} node={row['node_hostname']}"
    )
    print(
        f"scheduler={row.get('scheduler_name') or scheduler_name} "
        f"policy={row.get('policy_name') or policy_name} "
        f"backend={row.get('backend_name') or 'ssh-systemd'} "
        f"retries={int(row.get('retry_attempts') or 0)}/{int(row.get('retry_max') or max_retries)}"
    )
    print(f"reason: {row['placement_reason']}")
    allocations = row.get("allocations") or []
    if allocations:
        alloc_brief = ", ".join(
            f"{a.get('node_hostname')}:g{float(a.get('gpus', 0.0)):.0f}/c{float(a.get('cpus', 0.0)):.1f}"
            for a in allocations
        )
        print(f"allocations: {alloc_brief}")
    print(f"log: {row['log_path']}")


def load_job_config(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("job config must be a YAML mapping")
    return payload


def build_submission_policy(args: argparse.Namespace) -> StaticSubmissionPolicy:
    if args.policy_config:
        policy = load_submission_policy(args.policy_config)
    else:
        policy = StaticSubmissionPolicy()

    if args.max_active_jobs_per_user is not None:
        policy.max_active_jobs_per_user = int(args.max_active_jobs_per_user)
    if args.max_cpus_per_user is not None:
        policy.max_cpus_per_user = float(args.max_cpus_per_user)
    if args.max_gpus_per_user is not None:
        policy.max_gpus_per_user = float(args.max_gpus_per_user)
    if args.allow_host:
        policy.allowed_hosts = {str(v).strip().lower() for v in args.allow_host if v}
    if args.deny_host:
        policy.denied_hosts = {str(v).strip().lower() for v in args.deny_host if v}
    return policy


def print_overview(nodes, reservations, gpu_index_reservations) -> None:
    if not nodes:
        print("No registered machines found.")
        return

    header = (
        f"{'HOST':<20} {'IP':<15} {'STATUS':<12} {'CPU%':>6} {'CPU free est':>12} "
        f"{'Mem free GB':>11} {'GPU free est':>12} {'GPU util%':>9} {'GPU idx free':<14} "
        f"{'GPU idx resv':<14} {'GPU users':<20}"
    )
    print(header)
    print("-" * len(header))

    total_cpus = 0.0
    total_gpus = 0.0
    for node in nodes:
        probe_issue = _overview_probe_issue(node)
        gpu_users = ",".join(node.gpu_users[:2]) if not probe_issue else ""
        total_cpus += node.cpus_total
        total_gpus += node.gpus_total
        if probe_issue:
            print(
                f"{node.hostname[:20]:<20} {node.ip[:15]:<15} {'probe-error':<12} "
                f"{'n/a':>6} {'n/a':>12} {'n/a':>11} {'n/a':>12} {'n/a':>9} {'n/a':<14} "
                f"{'n/a':<14} {'':<20}"
            )
            print(f"  note: {probe_issue}")
            continue

        cpu_free_est, gpu_free_est = format_capacity_brief(node, reservations)
        free_idx = _format_gpu_indices(_schedulable_gpu_indices(node, gpu_index_reservations))
        reserved_idx = _format_gpu_indices(
            sorted(
                int(idx)
                for idx in (gpu_index_reservations.get(str(node.node_id), set()) or set())
            )
        )
        print(
            f"{node.hostname[:20]:<20} {node.ip[:15]:<15} {'ok':<12} {node.cpu_percent:>6.1f} "
            f"{cpu_free_est:>12.1f} {node.memory_available_gb:>11.1f} "
            f"{gpu_free_est:>12.1f} {node.gpu_util_avg:>9.1f} {free_idx:<14} {reserved_idx:<14} {gpu_users:<20}"
        )

    print()
    print(f"nodes={len(nodes)} total_cpus={total_cpus:.1f} total_gpus={total_gpus:.1f}")


def _overview_probe_issue(node) -> str:
    extras = getattr(node, "extras", {}) or {}
    if not isinstance(extras, dict):
        return ""
    if extras.get("probe_timeout"):
        return "probe timed out"
    for key in ("probe_error", "probe_wait_error"):
        value = extras.get(key)
        if value:
            return str(value)
    return ""


def _schedulable_gpu_indices(node, gpu_index_reservations: dict[str, set[int]]) -> list[int]:
    extras = getattr(node, "extras", {}) or {}
    gpu_details = extras.get("gpu_details", []) if isinstance(extras, dict) else []
    reserved = {
        int(idx)
        for idx in (gpu_index_reservations.get(str(getattr(node, "node_id", "")), set()) or set())
    }
    if isinstance(gpu_details, list) and gpu_details:
        free: list[int] = []
        for item in gpu_details:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("index")
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if idx in reserved:
                continue
            if float(item.get("proc_count", 0.0) or 0.0) > 0.0:
                continue
            free.append(idx)
        return sorted(free)
    return []


def _format_gpu_indices(indices: list[int]) -> str:
    if not indices:
        return "-"
    return ",".join(str(idx) for idx in indices)


def print_doctor(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No registered machines found.")
        return
    header = (
        f"{'HOST':<20} {'MODE':<8} {'SSH':<8} {'PYTHON':<8} {'SYSTEMD':<8} {'PROBE':<8} DETAILS"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ssh_ok = _fmt_ok(row.get("ssh_ok"))
        py_ok = _fmt_ok(row.get("python_ok"))
        systemd_ok = _fmt_ok(row.get("systemd_ok"))
        probe_ok = _fmt_ok(row.get("probe_ok"))
        detail = (
            row.get("probe_detail")
            or row.get("systemd_detail")
            or row.get("python_detail")
            or row.get("ssh_detail")
            or ""
        )
        print(
            f"{str(row.get('host') or '')[:20]:<20} {str(row.get('mode') or ''):<8} "
            f"{ssh_ok:<8} {py_ok:<8} {systemd_ok:<8} {probe_ok:<8} {str(detail)[:120]}"
        )


def _fmt_ok(value: Any) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "fail"
    return "-"


def print_jobs(rows) -> None:
    if not rows:
        print("No jobs in DB.")
        return

    header = (
        f"{'JOB ID':<12} {'STATUS':<10} {'MODE':<11} {'SCHED':<10} {'NAME':<20} "
        f"{'NODE':<16} {'CPUs':>6} {'GPUs':>6} {'TRY':>7} {'CREATED':<20}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['job_id']:<12} {row['status']:<10} {(row.get('job_mode') or 'single'):<11} "
            f"{(row.get('scheduler_name') or 'balanced')[:10]:<10} {row['name'][:20]:<20} "
            f"{(row.get('node_hostname') or '-')[:16]:<16} {float(row['requested_cpus']):>6.1f} "
            f"{float(row['requested_gpus']):>6.1f} "
            f"{int(row.get('retry_attempts') or 0):>2d}/{int(row.get('retry_max') or 0):<4d} "
            f"{row['created_at'][:19]:<20}"
        )


def print_job_status(row) -> None:
    fields = [
        "job_id",
        "name",
        "status",
        "job_mode",
        "scheduler_name",
        "policy_name",
        "backend_name",
        "submit_user",
        "node_hostname",
        "node_ip",
        "requested_cpus",
        "requested_gpus",
        "retry_max",
        "retry_backoff_seconds",
        "retry_attempts",
        "created_at",
        "started_at",
        "ended_at",
        "return_code",
        "command",
        "log_path",
        "state_path",
        "error_text",
    ]
    for key in fields:
        print(f"{key}: {row.get(key)}")
    metadata_raw = row.get("metadata_json")
    if metadata_raw:
        try:
            parsed = yaml.safe_load(str(metadata_raw))
            metadata_compact = yaml.safe_dump(
                parsed, default_flow_style=False, sort_keys=True
            ).strip()
            print("metadata:")
            for line in metadata_compact.splitlines():
                print(f"  {line}")
        except Exception:
            print(f"metadata_json: {metadata_raw}")
    allocations = row.get("allocations") or []
    if allocations:
        print("allocations:")
        for alloc in allocations:
            print(
                f"  - {alloc.get('node_hostname')} ({alloc.get('node_ip')}): "
                f"cpus={float(alloc.get('cpus', 0.0)):.2f}, gpus={float(alloc.get('gpus', 0.0)):.2f}"
            )


def follow_logs(
    orch: Orchestrator, job_id: str, n_lines: int, poll_seconds: float
) -> None:
    sources = orch.log_sources(job_id)
    status_row = orch.status(job_id, refresh=True)
    saw_any_output = _emit_sources_from_files(
        sources=sources,
        n_lines=n_lines,
        full_history=(str(status_row.get("status")) == "FAILED"),
    )

    offsets: dict[str, int] = {}
    for source in sources:
        path = Path(source["path"])
        offsets[source["path"]] = path.stat().st_size if path.exists() else 0

    terminal = {"SUCCEEDED", "FAILED", "CANCELLED"}
    while True:
        time.sleep(max(0.2, poll_seconds))
        status_row = orch.status(job_id, refresh=True)
        saw_new = False
        for source in sources:
            source_path = source["path"]
            path = Path(source_path)
            if not path.exists():
                continue
            previous = offsets.get(source_path, 0)
            size = path.stat().st_size
            if size < previous:
                previous = 0
            if size > previous:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(previous)
                    chunk = handle.read()
                offsets[source_path] = size
                if chunk:
                    _emit_source_chunk(
                        source=source,
                        chunk=chunk,
                        show_label=_should_print_source_labels(sources),
                    )
                    saw_new = True
                    saw_any_output = True

        if str(status_row.get("status")) in terminal and not saw_new:
            late_new = False
            settle_deadline = time.time() + (5.0 if not saw_any_output else 2.0)
            while time.time() < settle_deadline:
                time.sleep(0.25)
                round_new = False
                for source in sources:
                    source_path = source["path"]
                    path = Path(source_path)
                    if not path.exists():
                        continue
                    previous = offsets.get(source_path, 0)
                    size = path.stat().st_size
                    if size > previous:
                        with path.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(previous)
                            chunk = handle.read()
                        offsets[source_path] = size
                        if chunk:
                            _emit_source_chunk(
                                source=source,
                                chunk=chunk,
                                show_label=_should_print_source_labels(sources),
                            )
                            round_new = True
                            saw_any_output = True
                if round_new:
                    late_new = True
            if not late_new:
                break


def _emit_sources_from_files(
    sources: list[dict[str, str]], n_lines: int, full_history: bool = False
) -> bool:
    show_labels = _should_print_source_labels(sources)
    emitted = False
    for source in sources:
        path = Path(source["path"])
        chunk = _read_source_file(path, n_lines=None if full_history else n_lines)
        if not chunk:
            continue
        _emit_source_chunk(source=source, chunk=chunk, show_label=show_labels)
        emitted = True
    return emitted


def _emit_source_chunk(source: dict[str, str], chunk: str, show_label: bool) -> None:
    stream = _stream_for_source(source)
    if show_label:
        stream.write(f"===== {source['label']} =====\n")
    stream.write(chunk)
    if not chunk.endswith("\n"):
        stream.write("\n")
    stream.flush()


def _stream_for_source(source: dict[str, str]):
    return sys.stderr if str(source.get("stream") or "stdout") == "stderr" else sys.stdout


def _should_print_source_labels(sources: list[dict[str, str]]) -> bool:
    return any("rank=" in str(source.get("label") or "") for source in sources)


def _read_source_file(path: Path, n_lines: int | None) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if n_lines is None:
            return handle.read()
        lines = handle.readlines()
    return "".join(lines[-max(1, int(n_lines)) :])


def _build_orchestrator(args: argparse.Namespace) -> Orchestrator:
    return Orchestrator(
        cluster_config=args.cluster_config,
        db_path=args.db,
        logs_dir=args.logs_dir,
        state_dir=args.state_dir,
    )


def apply_runtime_defaults(args: argparse.Namespace) -> None:
    config = load_app_config(args.config_file)
    defaults = config.as_dict() if config is not None else {}

    args.cluster_config = _resolve_path_value(
        cli_value=args.cluster_config,
        env_name="LAB_ORCH_CLUSTER_CONFIG",
        config_value=defaults.get("cluster_config"),
        fallback=str(Path.home() / ".lab_orch" / "cluster.yaml"),
    )
    args.db = _resolve_path_value(
        cli_value=args.db,
        env_name="LAB_ORCH_DB",
        config_value=defaults.get("db"),
        fallback=str(Path.home() / ".lab_orch" / "jobs.db"),
    )
    args.logs_dir = _resolve_path_value(
        cli_value=args.logs_dir,
        env_name="LAB_ORCH_LOGS",
        config_value=defaults.get("logs_dir"),
        fallback=str(Path.home() / ".lab_orch" / "logs"),
    )
    args.state_dir = _resolve_path_value(
        cli_value=args.state_dir,
        env_name="LAB_ORCH_STATE_DIR",
        config_value=defaults.get("state_dir"),
        fallback=str(Path.home() / ".lab_orch" / "state"),
    )


def _resolve_path_value(
    cli_value: str | None, env_name: str, config_value: str | None, fallback: str
) -> str:
    if cli_value:
        return str(Path(cli_value).expanduser())
    env_value = os.environ.get(env_name)
    if env_value:
        return str(Path(env_value).expanduser())
    if config_value:
        return str(Path(config_value).expanduser())
    return str(Path(fallback).expanduser())


if __name__ == "__main__":
    main()
