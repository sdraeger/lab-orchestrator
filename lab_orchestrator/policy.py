from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from .models import JobRequest, NodeSnapshot


class SubmissionPolicy(Protocol):
    name: str

    def validate_submit(
        self, request: JobRequest, submit_user: str, active_usage: dict[str, float]
    ) -> None: ...

    def filter_nodes(self, nodes: list[NodeSnapshot]) -> list[NodeSnapshot]: ...


@dataclass(slots=True)
class StaticSubmissionPolicy:
    max_active_jobs_per_user: int | None = None
    max_cpus_per_user: float | None = None
    max_gpus_per_user: float | None = None
    allowed_hosts: set[str] = field(default_factory=set)
    denied_hosts: set[str] = field(default_factory=set)
    name: str = "static"

    def validate_submit(
        self, request: JobRequest, submit_user: str, active_usage: dict[str, float]
    ) -> None:
        _ = submit_user
        jobs = int(active_usage.get("jobs", 0))
        cpus = float(active_usage.get("cpus", 0.0))
        gpus = float(active_usage.get("gpus", 0.0))

        if self.max_active_jobs_per_user is not None and jobs >= int(
            self.max_active_jobs_per_user
        ):
            raise RuntimeError(
                "Submission rejected by policy: active job limit reached "
                f"(limit={self.max_active_jobs_per_user}, active={jobs})."
            )

        if self.max_cpus_per_user is not None:
            projected_cpus = cpus + float(request.cpus)
            if projected_cpus > float(self.max_cpus_per_user) + 1e-9:
                raise RuntimeError(
                    "Submission rejected by policy: CPU quota exceeded "
                    f"(limit={self.max_cpus_per_user:.2f}, projected={projected_cpus:.2f})."
                )

        if self.max_gpus_per_user is not None:
            projected_gpus = gpus + float(request.gpus)
            if projected_gpus > float(self.max_gpus_per_user) + 1e-9:
                raise RuntimeError(
                    "Submission rejected by policy: GPU quota exceeded "
                    f"(limit={self.max_gpus_per_user:.2f}, projected={projected_gpus:.2f})."
                )

    def filter_nodes(self, nodes: list[NodeSnapshot]) -> list[NodeSnapshot]:
        if not self.allowed_hosts and not self.denied_hosts:
            return nodes
        filtered: list[NodeSnapshot] = []
        for node in nodes:
            identities = {
                str(node.hostname).strip().lower(),
                str(node.ip).strip().lower(),
                str(node.node_id).strip().lower(),
            }

            if self.allowed_hosts and not (identities & self.allowed_hosts):
                continue
            if self.denied_hosts and (identities & self.denied_hosts):
                continue
            filtered.append(node)
        return filtered

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "StaticSubmissionPolicy":
        return cls(
            max_active_jobs_per_user=_as_optional_int(
                payload.get("max_active_jobs_per_user")
            ),
            max_cpus_per_user=_as_optional_float(payload.get("max_cpus_per_user")),
            max_gpus_per_user=_as_optional_float(payload.get("max_gpus_per_user")),
            allowed_hosts=_normalize_hosts(payload.get("allowed_hosts")),
            denied_hosts=_normalize_hosts(payload.get("denied_hosts")),
            name=str(payload.get("name", "static")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_active_jobs_per_user": self.max_active_jobs_per_user,
            "max_cpus_per_user": self.max_cpus_per_user,
            "max_gpus_per_user": self.max_gpus_per_user,
            "allowed_hosts": sorted(self.allowed_hosts),
            "denied_hosts": sorted(self.denied_hosts),
        }


def load_submission_policy(path: str | Path) -> StaticSubmissionPolicy:
    file_path = Path(path).expanduser()
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if payload is None:
        return StaticSubmissionPolicy()
    if not isinstance(payload, dict):
        raise ValueError("policy config must be a YAML mapping")
    return StaticSubmissionPolicy.from_mapping(payload)


def _normalize_hosts(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError("policy field for hosts must be a YAML list")
    out: set[str] = set()
    for item in value:
        normalized = str(item).strip().lower()
        if normalized:
            out.add(normalized)
    return out


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    if out < 0:
        raise ValueError("policy integer limits must be >= 0")
    return out


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    out = float(value)
    if out < 0:
        raise ValueError("policy float limits must be >= 0")
    return out
