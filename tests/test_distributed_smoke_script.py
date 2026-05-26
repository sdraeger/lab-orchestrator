from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from lab_orchestrator.vgpu import VirtualGpu, VirtualGpuContext


def _load_distributed_smoke_module():
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "distributed_smoke.py"
    spec = importlib.util.spec_from_file_location("distributed_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distributed_smoke_script_single_process_mode() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "distributed_smoke.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--label", "dist-smoke"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["label"] == "dist-smoke"
    assert payload["world_size"] == 1
    assert payload["ok"] is True


def test_distributed_smoke_script_require_distributed_fails_without_world() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "distributed_smoke.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--require-distributed"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout.strip())
    assert payload["ok"] is False
    assert "WORLD_SIZE <= 1" in result.stderr


def test_distributed_smoke_script_reads_virtual_gpu_manifest() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "distributed_smoke.py"
    env = {
        "WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "0",
        "NODE_RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "23456",
        "LAB_ORCH_VGPU": "1",
        "LAB_ORCH_VGPU_COUNT": "2",
        "LAB_ORCH_VGPU_IDS": "0,1",
        "LAB_ORCH_VGPU_INDEX": "1",
        "LAB_ORCH_VGPU_MANIFEST_JSON": json.dumps(
            [
                {
                    "vgpu_index": 0,
                    "host": "node33",
                    "node_id": "node33",
                    "node_hostname": "node33",
                    "node_ip": "10.0.0.33",
                    "node_rank": 0,
                    "local_rank": 0,
                    "physical_gpu": 0,
                    "cuda_visible_devices": "0",
                },
                {
                    "vgpu_index": 1,
                    "host": "node34",
                    "node_id": "node34",
                    "node_hostname": "node34",
                    "node_ip": "10.0.0.34",
                    "node_rank": 1,
                    "local_rank": 0,
                    "physical_gpu": 1,
                    "cuda_visible_devices": "1",
                },
            ]
        ),
    }

    result = subprocess.run(
        [sys.executable, str(script_path), "--require-vgpu"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout.strip())
    assert payload["vgpu_enabled"] is True
    assert payload["vgpu_count"] == 2
    assert payload["vgpu_ids"] == [0, 1]
    assert payload["vgpu_index"] == 1


def test_backend_auto_prefers_gloo_for_multi_host_virtual_pool() -> None:
    module = _load_distributed_smoke_module()

    class _FakeDistributed:
        @staticmethod
        def is_nccl_available() -> bool:
            return True

    class _FakeTorch:
        distributed = _FakeDistributed()

    vgpu = VirtualGpuContext(
        count=2,
        ids=[0, 1],
        current_index=0,
        current=None,
        gpus=[
            VirtualGpu(
                vgpu_index=0,
                host="node33",
                node_id="node33",
                node_hostname="node33",
                node_ip="10.0.0.33",
                node_rank=0,
                local_rank=0,
                physical_gpu=0,
                cuda_visible_devices="0",
            ),
            VirtualGpu(
                vgpu_index=1,
                host="node34",
                node_id="node34",
                node_hostname="node34",
                node_ip="10.0.0.34",
                node_rank=1,
                local_rank=0,
                physical_gpu=0,
                cuda_visible_devices="0",
            ),
        ],
    )
    args = argparse.Namespace(backend="auto")

    assert module._backend_for(args, _FakeTorch(), True, vgpu) == "gloo"
