from __future__ import annotations

import json

from lab_orchestrator.vgpu import load_virtual_gpu_context


def test_load_virtual_gpu_context() -> None:
    env = {
        "LAB_ORCH_VGPU_COUNT": "3",
        "LAB_ORCH_VGPU_IDS": "0,1,2",
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
                    "cuda_visible_devices": "0,1",
                },
                {
                    "vgpu_index": 1,
                    "host": "node33",
                    "node_id": "node33",
                    "node_hostname": "node33",
                    "node_ip": "10.0.0.33",
                    "node_rank": 0,
                    "local_rank": 1,
                    "physical_gpu": 1,
                    "cuda_visible_devices": "0,1",
                },
                {
                    "vgpu_index": 2,
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

    ctx = load_virtual_gpu_context(env)
    assert ctx is not None
    assert ctx.count == 3
    assert ctx.ids == [0, 1, 2]
    assert ctx.current_index == 1
    assert ctx.current is not None
    assert ctx.current.node_hostname == "node33"
    assert ctx.current.physical_gpu == 1
