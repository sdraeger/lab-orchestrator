from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from .bootstrap import bootstrap_cluster, stop_cluster
from .models import JobRequest
from .orchestrator import Orchestrator
from .scheduler import format_capacity_brief
from .utils import parse_env_pairs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "cmd", None):
        parser.print_help()
        return

    try:
        if args.cmd == "bootstrap":
            bootstrap_cluster(args.config, dry_run=args.dry_run)
            print("Bootstrap complete.")
            return
        if args.cmd == "stop-cluster":
            stop_cluster(args.config, dry_run=args.dry_run)
            print("Cluster stop complete.")
            return

        orch = Orchestrator(
            ray_address=resolve_ray_address(args.ray_address, args.cluster_config),
            namespace=args.namespace,
            db_path=args.db,
            logs_dir=args.logs_dir,
        )

        if args.cmd == "overview":
            payload = orch.overview()
            print_overview(payload["nodes"], payload["reservations"])
            return

        if args.cmd == "submit":
            cfg = load_job_config(args.job_config) if args.job_config else {}
            env_from_cfg = cfg.get("env", {})
            if env_from_cfg is None:
                env_from_cfg = {}
            if not isinstance(env_from_cfg, dict):
                raise ValueError(
                    "submit config field 'env' must be a mapping of KEY: VALUE"
                )

            name = args.name or cfg.get("name")
            command = args.command or cfg.get("command")
            cpus = args.cpus if args.cpus is not None else float(cfg.get("cpus", 1.0))
            gpus = args.gpus if args.gpus is not None else float(cfg.get("gpus", 0.0))
            workdir = args.workdir or cfg.get("workdir") or os.getcwd()
            distributed_cfg = bool(cfg.get("distributed", False))
            distributed = bool(args.distributed or distributed_cfg)

            if not name:
                raise ValueError("submit requires --name (or a 'name' in --job-config)")
            if not command:
                raise ValueError(
                    "submit requires --command (or a 'command' in --job-config)"
                )

            merged_env = {str(k): str(v) for k, v in env_from_cfg.items()}
            merged_env.update(parse_env_pairs(args.env))

            request = JobRequest(
                name=str(name),
                command=str(command),
                cpus=float(cpus),
                gpus=float(gpus),
                workdir=str(workdir),
                env=merged_env,
                distributed=distributed,
            )
            row = orch.submit(request)
            print(
                f"submitted job_id={row['job_id']} status={row['status']} "
                f"mode={row.get('job_mode', 'single')} node={row['node_hostname']}"
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
        prog="lab-orch",
        description="Ray-based job orchestration for lab cluster machines",
    )

    parser.add_argument(
        "--ray-address",
        default=os.environ.get("LAB_ORCH_RAY_ADDRESS", "auto"),
        help="Ray cluster address (default: auto)",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("LAB_ORCH_NAMESPACE", "lab-orchestrator"),
        help="Ray namespace used for detached actors",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get(
            "LAB_ORCH_DB", str(Path.home() / ".lab_orch" / "jobs.db")
        ),
        help="SQLite DB path",
    )
    parser.add_argument(
        "--logs-dir",
        default=os.environ.get(
            "LAB_ORCH_LOGS", str(Path.home() / ".lab_orch" / "logs")
        ),
        help="Directory for job logs",
    )
    parser.add_argument(
        "--cluster-config",
        default=os.environ.get("LAB_ORCH_CLUSTER_CONFIG"),
        help="Optional YAML cluster config used to resolve --ray-address auto to head_address:ray_port",
    )

    sub = parser.add_subparsers(dest="cmd")

    p_bootstrap = sub.add_parser(
        "bootstrap", help="Start/attach Ray on head/worker hosts via SSH"
    )
    p_bootstrap.add_argument("--config", required=True, help="YAML config path")
    p_bootstrap.add_argument("--dry-run", action="store_true")

    p_stop = sub.add_parser(
        "stop-cluster", help="Stop Ray on all hosts from bootstrap config"
    )
    p_stop.add_argument("--config", required=True, help="YAML config path")
    p_stop.add_argument("--dry-run", action="store_true")

    sub.add_parser("overview", help="Show cluster-wide node resources and load")

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
    p_submit.add_argument("--workdir", help="Command working directory")
    p_submit.add_argument(
        "--distributed",
        action="store_true",
        help="Force distributed mode (one process per GPU across nodes)",
    )
    p_submit.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE format (repeatable)",
    )

    p_jobs = sub.add_parser("jobs", help="List jobs")
    p_jobs.add_argument("--limit", type=int, default=50)
    p_jobs.add_argument(
        "--refresh", action="store_true", help="Refresh running status from Ray"
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


def load_job_config(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("job config must be a YAML mapping")
    return payload


def load_cluster_config(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("cluster config must be a YAML mapping")
    return payload


def resolve_ray_address(ray_address: str, cluster_config: str | None) -> str:
    if ray_address != "auto":
        return ray_address
    cfg_path = _pick_cluster_config_path(cluster_config)
    if cfg_path is None:
        return ray_address
    cfg = load_cluster_config(cfg_path)
    head = str(cfg.get("head_address", cfg.get("head", ""))).strip()
    port = int(cfg.get("ray_port", 6379))
    if not head:
        return ray_address
    return f"{head}:{port}"


def _pick_cluster_config_path(user_path: str | None) -> str | None:
    if user_path:
        p = Path(user_path).expanduser()
        return str(p) if p.exists() else None
    candidates = [
        Path.cwd() / "cluster.yaml",
        Path.cwd() / "cluster.example.yaml",
        Path.home() / "lab-orchestrator" / "cluster.yaml",
        Path.home() / "lab-orchestrator" / "cluster.example.yaml",
        Path.home() / ".lab_orch" / "cluster.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def print_overview(nodes, reservations) -> None:
    if not nodes:
        print("No live Ray nodes found.")
        return

    header = (
        f"{'HOST':<20} {'IP':<15} {'CPU%':>6} {'CPU free est':>12} "
        f"{'Mem free GB':>11} {'GPU free est':>12} {'GPU util%':>9} {'GPU users':<20}"
    )
    print(header)
    print("-" * len(header))

    total_cpus = 0.0
    total_gpus = 0.0
    for node in nodes:
        cpu_free_est, gpu_free_est = format_capacity_brief(node, reservations)
        gpu_users = ",".join(node.gpu_users[:2])
        total_cpus += node.cpus_total
        total_gpus += node.gpus_total
        print(
            f"{node.hostname[:20]:<20} {node.ip[:15]:<15} {node.cpu_percent:>6.1f} "
            f"{cpu_free_est:>12.1f} {node.memory_available_gb:>11.1f} "
            f"{gpu_free_est:>12.1f} {node.gpu_util_avg:>9.1f} {gpu_users:<20}"
        )

    print()
    print(f"nodes={len(nodes)} total_cpus={total_cpus:.1f} total_gpus={total_gpus:.1f}")


def print_jobs(rows) -> None:
    if not rows:
        print("No jobs in DB.")
        return

    header = (
        f"{'JOB ID':<12} {'STATUS':<10} {'MODE':<11} {'NAME':<20} "
        f"{'NODE':<16} {'CPUs':>6} {'GPUs':>6} {'CREATED':<20}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['job_id']:<12} {row['status']:<10} {(row.get('job_mode') or 'single'):<11} {row['name'][:20]:<20} "
            f"{(row.get('node_hostname') or '-')[:16]:<16} {float(row['requested_cpus']):>6.1f} "
            f"{float(row['requested_gpus']):>6.1f} {row['created_at'][:19]:<20}"
        )


def print_job_status(row) -> None:
    fields = [
        "job_id",
        "name",
        "status",
        "job_mode",
        "submit_user",
        "node_hostname",
        "node_ip",
        "requested_cpus",
        "requested_gpus",
        "created_at",
        "started_at",
        "ended_at",
        "return_code",
        "command",
        "log_path",
        "error_text",
    ]
    for key in fields:
        print(f"{key}: {row.get(key)}")
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
    status_row = orch.status(job_id, refresh=True)
    if str(status_row.get("status")) == "FAILED":
        print(orch.logs_all(job_id), end="")
        return

    sources = orch.log_sources(job_id)
    initial = _render_sources_from_files(sources=sources, n_lines=n_lines)
    if initial:
        print(initial, end="")

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
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(previous)
                    chunk = f.read()
                offsets[source_path] = size
                if chunk:
                    if len(sources) > 1:
                        print(f"===== {source['label']} =====")
                    print(chunk, end="")
                    saw_new = True

        if str(status_row.get("status")) in terminal and not saw_new:
            # Drain terminal tail with a small grace window to catch late flushes.
            late_new = False
            for _ in range(8):
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
                        with path.open("r", encoding="utf-8", errors="replace") as f:
                            f.seek(previous)
                            chunk = f.read()
                        offsets[source_path] = size
                        if chunk:
                            if len(sources) > 1:
                                print(f"===== {source['label']} =====")
                            print(chunk, end="")
                            round_new = True
                if round_new:
                    late_new = True
                else:
                    break
            if late_new:
                continue
            if str(status_row.get("status")) == "FAILED":
                # Guarantee full failed logs, independent of stream timing.
                print(orch.logs_all(job_id), end="")
            break


def _render_sources_from_files(sources: list[dict[str, str]], n_lines: int) -> str:
    chunks: list[str] = []
    multi = len(sources) > 1
    for source in sources:
        path = Path(source["path"])
        text = _read_tail_from_file(path=path, n_lines=n_lines)
        if not text:
            continue
        if multi:
            chunks.append(f"===== {source['label']} =====\n")
        chunks.append(text)
        if not text.endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)


def _read_tail_from_file(path: Path, n_lines: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max(1, int(n_lines)) :])


if __name__ == "__main__":
    main()
