from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class VirtualGpu:
    vgpu_index: int
    host: str
    node_id: str
    node_hostname: str
    node_ip: str
    node_rank: int
    local_rank: int
    physical_gpu: int
    cuda_visible_devices: str


@dataclass(slots=True)
class VirtualGpuContext:
    count: int
    ids: list[int]
    current_index: int | None
    current: VirtualGpu | None
    gpus: list[VirtualGpu]

    @property
    def enabled(self) -> bool:
        return self.count > 0


def load_virtual_gpu_context(
    environ: Mapping[str, str] | None = None,
) -> VirtualGpuContext | None:
    env = environ or os.environ
    raw_count = str(env.get("LAB_ORCH_VGPU_COUNT") or "").strip()
    raw_manifest = str(env.get("LAB_ORCH_VGPU_MANIFEST_JSON") or "").strip()
    if not raw_count and not raw_manifest:
        return None

    count = _int_or_default(raw_count, default=0)
    ids = _parse_ids(str(env.get("LAB_ORCH_VGPU_IDS") or ""))
    if not ids and count > 0:
        ids = list(range(count))

    manifest_payload: list[dict[str, Any]] = []
    if raw_manifest:
        try:
            parsed = json.loads(raw_manifest)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            manifest_payload = [item for item in parsed if isinstance(item, dict)]

    gpus = [
        VirtualGpu(
            vgpu_index=_int_or_default(item.get("vgpu_index"), default=0),
            host=str(item.get("host") or ""),
            node_id=str(item.get("node_id") or ""),
            node_hostname=str(item.get("node_hostname") or ""),
            node_ip=str(item.get("node_ip") or ""),
            node_rank=_int_or_default(item.get("node_rank"), default=0),
            local_rank=_int_or_default(item.get("local_rank"), default=0),
            physical_gpu=_int_or_default(item.get("physical_gpu"), default=0),
            cuda_visible_devices=str(item.get("cuda_visible_devices") or ""),
        )
        for item in manifest_payload
    ]
    if not ids and gpus:
        ids = [gpu.vgpu_index for gpu in gpus]
    if count <= 0:
        count = len(ids) or len(gpus)

    current_index = _int_or_none(env.get("LAB_ORCH_VGPU_INDEX"))
    current = None
    if current_index is not None:
        for gpu in gpus:
            if gpu.vgpu_index == current_index:
                current = gpu
                break

    return VirtualGpuContext(
        count=count,
        ids=ids,
        current_index=current_index,
        current=current,
        gpus=gpus,
    )


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _parse_ids(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out
