from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MachineSpec:
    host: str
    ssh_host: str | None = None
    ssh_user: str | None = None
    python_bin: str = "python3"
    local: bool = False
    labels: list[str] = field(default_factory=list)

    @property
    def display_host(self) -> str:
        return self.host

    @property
    def connect_host(self) -> str:
        return str(self.ssh_host or self.host)

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "ssh_host": self.ssh_host,
            "ssh_user": self.ssh_user,
            "python_bin": self.python_bin,
            "local": bool(self.local),
            "labels": list(self.labels),
        }

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any], default_ssh_user: str | None = None
    ) -> "MachineSpec":
        host = str(payload.get("host", "")).strip()
        if not host:
            raise ValueError("cluster machine entries require a non-empty 'host'")
        ssh_host_raw = payload.get("ssh_host")
        ssh_host = str(ssh_host_raw).strip() if ssh_host_raw not in {None, ""} else None
        ssh_user_raw = payload.get("ssh_user", default_ssh_user)
        ssh_user = (
            str(ssh_user_raw).strip() if ssh_user_raw not in {None, ""} else None
        )
        python_bin = str(payload.get("python_bin", "python3")).strip() or "python3"
        local = bool(payload.get("local", False))
        labels = payload.get("labels", [])
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise ValueError("cluster machine field 'labels' must be a YAML list")
        return cls(
            host=host,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            python_bin=python_bin,
            local=local,
            labels=[str(label).strip() for label in labels if str(label).strip()],
        )


@dataclass(slots=True)
class ClusterRegistry:
    machines: list[MachineSpec] = field(default_factory=list)
    ssh_user: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ssh_user": self.ssh_user,
            "machines": [machine.as_dict() for machine in self.machines],
        }

    def get_machine(self, selector: str) -> MachineSpec | None:
        normalized = str(selector).strip().lower()
        for machine in self.machines:
            keys = {
                machine.host.strip().lower(),
                machine.connect_host.strip().lower(),
            }
            if normalized in keys:
                return machine
        return None

    def upsert_machine(self, machine: MachineSpec) -> None:
        existing = self.get_machine(machine.host)
        if existing is None:
            self.machines.append(machine)
        else:
            existing.ssh_host = machine.ssh_host
            existing.ssh_user = machine.ssh_user
            existing.python_bin = machine.python_bin
            existing.local = machine.local
            existing.labels = list(machine.labels)
        self.machines.sort(key=lambda item: item.host.lower())

    def remove_machine(self, selector: str) -> MachineSpec:
        normalized = str(selector).strip().lower()
        for idx, machine in enumerate(self.machines):
            keys = {
                machine.host.strip().lower(),
                machine.connect_host.strip().lower(),
            }
            if normalized in keys:
                return self.machines.pop(idx)
        raise RuntimeError(f"Unknown registered machine '{selector}'")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ClusterRegistry":
        ssh_user_raw = payload.get("ssh_user")
        ssh_user = (
            str(ssh_user_raw).strip() if ssh_user_raw not in {None, ""} else None
        )
        machines_payload = payload.get("machines", [])
        if machines_payload is None:
            machines_payload = []
        if not isinstance(machines_payload, list):
            raise ValueError("cluster config field 'machines' must be a YAML list")
        machines = [
            MachineSpec.from_mapping(item, default_ssh_user=ssh_user)
            for item in machines_payload
        ]
        registry = cls(machines=machines, ssh_user=ssh_user)
        registry.machines.sort(key=lambda item: item.host.lower())
        return registry


def load_cluster_registry(path: str | Path) -> ClusterRegistry:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return ClusterRegistry()
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if payload is None:
        return ClusterRegistry()
    if not isinstance(payload, dict):
        raise ValueError("cluster config must be a YAML mapping")
    return ClusterRegistry.from_mapping(payload)


def save_cluster_registry(path: str | Path, registry: ClusterRegistry) -> None:
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(registry.as_dict(), sort_keys=False)
    file_path.write_text(text, encoding="utf-8")
