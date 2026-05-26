#!/usr/bin/env python3
from __future__ import annotations

__test__ = False

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small smoke test for lab-orch remote execution"
    )
    parser.add_argument(
        "--label",
        default="orch-smoke",
        help="Free-form label included in the output payload",
    )
    parser.add_argument(
        "--expect-host",
        help="Fail if socket.gethostname() does not match this value",
    )
    parser.add_argument(
        "--expect-visible-gpus",
        help="Fail if CUDA_VISIBLE_DEVICES does not match this exact string",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if torch.cuda.is_available() is false",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay before exit so you can inspect status/log streaming",
    )
    parser.add_argument(
        "--output-json",
        help="Optional file path where the JSON payload should also be written",
    )
    return parser


def _load_torch() -> tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return torch, None


def collect_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    payload: dict[str, Any] = {
        "label": str(args.label),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "argv": sys.argv[1:],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lab_orch_job_id": os.environ.get("LAB_ORCH_JOB_ID"),
        "lab_orch_distributed": os.environ.get("LAB_ORCH_DISTRIBUTED"),
    }
    errors: list[str] = []

    if args.expect_host and payload["hostname"] != str(args.expect_host):
        errors.append(
            f"expected hostname {args.expect_host!r}, got {payload['hostname']!r}"
        )
    if (
        args.expect_visible_gpus is not None
        and payload["cuda_visible_devices"] != str(args.expect_visible_gpus)
    ):
        errors.append(
            "expected CUDA_VISIBLE_DEVICES "
            f"{args.expect_visible_gpus!r}, got {payload['cuda_visible_devices']!r}"
        )

    torch_mod, torch_error = _load_torch()
    payload["torch_import_ok"] = torch_mod is not None
    payload["torch_import_error"] = torch_error
    if torch_mod is not None:
        torch = torch_mod
        payload["torch_version"] = str(torch.__version__)
        payload["cpu_sum"] = float(torch.arange(4, dtype=torch.float32).sum().item())
        cuda_ok = bool(torch.cuda.is_available())
        payload["cuda_available"] = cuda_ok
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_ok else 0
        if cuda_ok:
            current_device = torch.device("cuda:0")
            payload["cuda_sum"] = float(
                torch.arange(4, dtype=torch.float32, device=current_device).sum().item()
            )
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        else:
            payload["cuda_sum"] = None
            payload["cuda_device_name"] = None
        if args.require_cuda and not cuda_ok:
            errors.append("CUDA was required but torch.cuda.is_available() is false")
    else:
        payload["torch_version"] = None
        payload["cpu_sum"] = None
        payload["cuda_available"] = False
        payload["cuda_device_count"] = 0
        payload["cuda_sum"] = None
        payload["cuda_device_name"] = None
        if args.require_cuda:
            errors.append(
                "CUDA was required but torch could not be imported: "
                f"{torch_error or 'unknown torch import error'}"
            )

    payload["ok"] = not errors
    payload["errors"] = errors
    return payload, errors


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload, errors = collect_report(args)

    text = json.dumps(payload, sort_keys=True)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if float(args.sleep_seconds) > 0:
        time.sleep(float(args.sleep_seconds))

    if errors:
        for item in errors:
            print(f"ERROR: {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
