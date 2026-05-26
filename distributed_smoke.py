#!/usr/bin/env python3
from __future__ import annotations

__test__ = False

import argparse
import datetime as dt
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from lab_orchestrator.vgpu import VirtualGpuContext, load_virtual_gpu_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distributed smoke test for lab-orch multi-node launch"
    )
    parser.add_argument(
        "--label",
        default="orch-distributed-smoke",
        help="Free-form label included in the output payload",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gloo", "nccl"],
        default="auto",
        help="torch.distributed backend to use",
    )
    parser.add_argument(
        "--require-distributed",
        action="store_true",
        help="Fail unless WORLD_SIZE > 1 and process group init succeeds",
    )
    parser.add_argument(
        "--require-vgpu",
        action="store_true",
        help="Fail unless the orchestrator virtual GPU manifest is present and contiguous",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA is available on each worker",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay before exit for status/log inspection",
    )
    parser.add_argument(
        "--init-timeout-seconds",
        type=float,
        default=30.0,
        help="Timeout for torch.distributed process-group initialization",
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path prefix; each rank writes PREFIX.rank<RANK>.json",
    )
    return parser


def _load_torch() -> tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return torch, None


def _backend_for(
    args: argparse.Namespace,
    torch: Any,
    cuda_ok: bool,
    vgpu: VirtualGpuContext | None,
) -> str:
    if args.backend != "auto":
        return str(args.backend)
    # For multi-host smoke validation, prefer gloo by default. This verifies
    # rendezvous and virtual GPU assignment without turning the periodic smoke
    # test into a cluster-specific NCCL configuration check.
    host_count = len(
        {
            gpu.node_hostname or gpu.host or gpu.node_id
            for gpu in (vgpu.gpus if vgpu is not None else [])
            if (gpu.node_hostname or gpu.host or gpu.node_id)
        }
    )
    if host_count > 1:
        return "gloo"
    if cuda_ok and hasattr(torch.distributed, "is_nccl_available") and torch.distributed.is_nccl_available():
        return "nccl"
    return "gloo"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def collect_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    payload: dict[str, Any] = {
        "label": str(args.label),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "pid": os.getpid(),
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": _int_env("WORLD_SIZE", 1),
        "rank": _int_env("RANK", 0),
        "local_rank": _int_env("LOCAL_RANK", 0),
        "node_rank": _int_env("NODE_RANK", 0),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "lab_orch_distributed": os.environ.get("LAB_ORCH_DISTRIBUTED"),
        "lab_orch_job_id": os.environ.get("LAB_ORCH_JOB_ID"),
    }
    errors: list[str] = []
    world_size = int(payload["world_size"])
    if args.require_distributed and world_size <= 1:
        errors.append("distributed mode was required but WORLD_SIZE <= 1")

    vgpu = load_virtual_gpu_context()
    payload["vgpu_enabled"] = bool(vgpu is not None and vgpu.enabled)
    payload["vgpu_count"] = int(vgpu.count) if vgpu is not None else 0
    payload["vgpu_ids"] = list(vgpu.ids) if vgpu is not None else []
    payload["vgpu_index"] = int(vgpu.current_index) if vgpu and vgpu.current_index is not None else None
    payload["vgpu_current_physical_gpu"] = (
        int(vgpu.current.physical_gpu) if vgpu and vgpu.current is not None else None
    )
    payload["vgpu_hosts"] = sorted(
        {gpu.node_hostname or gpu.host for gpu in (vgpu.gpus if vgpu else []) if (gpu.node_hostname or gpu.host)}
    )
    payload["vgpu_host_count"] = len(payload["vgpu_hosts"])
    payload["vgpu_manifest_size"] = len(vgpu.gpus) if vgpu is not None else 0
    if vgpu is not None and vgpu.enabled:
        expected_ids = list(range(vgpu.count))
        if vgpu.ids != expected_ids:
            errors.append(
                f"virtual GPU ids are not contiguous 0..N-1: got {vgpu.ids!r}, expected {expected_ids!r}"
            )
    if args.require_vgpu and not (vgpu is not None and vgpu.enabled):
        errors.append("virtual GPU manifest was required but not present")

    torch_mod, torch_error = _load_torch()
    payload["torch_import_ok"] = torch_mod is not None
    payload["torch_import_error"] = torch_error
    payload["distributed_init_ok"] = False
    payload["distributed_backend"] = None
    payload["all_reduce_sum"] = None
    payload["barrier_ok"] = False
    payload["cuda_local_sum"] = None
    payload["cuda_device_name"] = None

    if torch_mod is None:
        payload["torch_version"] = None
        payload["cuda_available"] = False
        payload["cuda_device_count"] = 0
        if args.require_cuda or world_size > 1:
            errors.append(
                "torch import failed: "
                f"{torch_error or 'unknown torch import error'}"
            )
        payload["ok"] = not errors
        payload["errors"] = errors
        return payload, errors

    torch = torch_mod
    payload["torch_version"] = str(torch.__version__)
    cuda_ok = bool(torch.cuda.is_available())
    payload["cuda_available"] = cuda_ok
    payload["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_ok else 0
    if args.require_cuda and not cuda_ok:
        errors.append("CUDA was required but torch.cuda.is_available() is false")
    if cuda_ok:
        assigned_cuda_index = int(payload["local_rank"]) if world_size > 1 else 0
        try:
            if assigned_cuda_index >= int(payload["cuda_device_count"]):
                raise RuntimeError(
                    "assigned local_rank exceeds visible CUDA device count: "
                    f"local_rank={assigned_cuda_index}, "
                    f"visible={int(payload['cuda_device_count'])}"
                )
            torch.cuda.set_device(assigned_cuda_index)
            payload["cuda_device_name"] = str(
                torch.cuda.get_device_name(assigned_cuda_index)
            )
            local_value = torch.arange(
                4, dtype=torch.float32, device=torch.device("cuda", assigned_cuda_index)
            )
            payload["cuda_local_sum"] = float(local_value.sum().item())
        except Exception as exc:
            errors.append(f"local CUDA check failed: {type(exc).__name__}: {exc}")

    should_init_distributed = world_size > 1 and (
        args.require_distributed or os.environ.get("LAB_ORCH_DISTRIBUTED") == "1"
    )
    if should_init_distributed:
        backend = _backend_for(args, torch, cuda_ok, vgpu)
        payload["distributed_backend"] = backend
        if backend == "nccl" and not cuda_ok:
            errors.append("selected nccl backend but CUDA is not available")
        else:
            try:
                if backend == "nccl":
                    tensor_device = torch.device("cuda", int(payload["local_rank"]))
                else:
                    tensor_device = torch.device("cpu")
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=world_size,
                    rank=int(payload["rank"]),
                    timeout=dt.timedelta(
                        seconds=max(1.0, float(args.init_timeout_seconds))
                    ),
                )
                payload["distributed_init_ok"] = True
                value = torch.tensor(
                    [float(int(payload["rank"]) + 1)],
                    dtype=torch.float32,
                    device=tensor_device,
                )
                torch.distributed.all_reduce(value)
                payload["all_reduce_sum"] = float(value.item())
                torch.distributed.barrier()
                payload["barrier_ok"] = True
            except Exception as exc:
                errors.append(f"torch.distributed failed: {type(exc).__name__}: {exc}")
            finally:
                try:
                    if torch.distributed.is_initialized():
                        torch.distributed.destroy_process_group()
                except Exception:
                    pass

    payload["ok"] = not errors
    payload["errors"] = errors
    return payload, errors


def _write_output(prefix: str, rank: int, text: str) -> None:
    output_path = Path(f"{prefix}.rank{rank}.json").expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload, errors = collect_report(args)

    text = json.dumps(payload, sort_keys=True)
    print(text)

    if args.output_json:
        _write_output(str(args.output_json), int(payload.get("rank", 0) or 0), text)

    if float(args.sleep_seconds) > 0:
        time.sleep(float(args.sleep_seconds))

    if errors:
        for item in errors:
            print(f"ERROR: {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
