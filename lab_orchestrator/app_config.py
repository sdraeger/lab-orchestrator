from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    cluster_config: str
    db: str
    logs_dir: str
    state_dir: str

    def as_dict(self) -> dict[str, str]:
        return {
            "cluster_config": self.cluster_config,
            "db": self.db,
            "logs_dir": self.logs_dir,
            "state_dir": self.state_dir,
        }


def default_config_file() -> Path:
    return Path.home() / ".lab_orch" / "config.yaml"


def load_app_config(path: str | Path) -> AppConfig | None:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return None
    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("app config must be a YAML mapping")
    return AppConfig(
        cluster_config=str(payload.get("cluster_config", "")).strip(),
        db=str(payload.get("db", "")).strip(),
        logs_dir=str(payload.get("logs_dir", "")).strip(),
        state_dir=str(payload.get("state_dir", "")).strip(),
    )


def save_app_config(path: str | Path, config: AppConfig) -> None:
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config.as_dict(), sort_keys=False)
    file_path.write_text(text, encoding="utf-8")
    _chmod_best_effort(file_path, 0o600)


def init_app_config(root: str | Path) -> AppConfig:
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    _chmod_best_effort(root_path, 0o700)
    logs_dir = root_path / "logs"
    state_dir = root_path / "state"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    _chmod_best_effort(logs_dir, 0o700)
    _chmod_best_effort(state_dir, 0o700)
    return AppConfig(
        cluster_config=str(root_path / "cluster.yaml"),
        db=str(root_path / "jobs.db"),
        logs_dir=str(logs_dir),
        state_dir=str(state_dir),
    )


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass
