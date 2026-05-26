from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from lab_orchestrator import cli
from lab_orchestrator.app_config import load_app_config
from lab_orchestrator.models import NodeSnapshot


def test_build_run_request_cpu_only() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["run", "--target-host", "node33", "--", "python", "script.py", "--x", "1"]
    )
    request, targets = cli.build_run_request(args)
    assert request.gpus == 0.0
    assert request.command == "python script.py --x 1"
    assert request.command_argv == ["python", "script.py", "--x", "1"]
    assert request.use_shell is False
    assert targets == ["node33"]


def test_build_run_request_auto_schedule_gpu() -> None:
    parser = cli.build_implicit_run_parser()
    args = parser.parse_args(["--gpus", "1", "python", "train.py", "--config", "conf.yaml"])
    request, targets = cli.build_run_request(args)
    assert request.gpus == 1.0
    assert request.command == "python train.py --config conf.yaml"
    assert request.command_argv == ["python", "train.py", "--config", "conf.yaml"]
    assert request.use_shell is False
    assert targets == []


def test_build_run_request_systemd_properties() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--runtime-max-seconds",
            "60",
            "--memory-max",
            "8G",
            "--tasks-max",
            "128",
            "--cpu-quota-percent",
            "250",
            "--",
            "python",
            "script.py",
        ]
    )
    request, _targets = cli.build_run_request(args)
    assert request.systemd_properties == {
        "RuntimeMaxSec": "60s",
        "MemoryMax": "8G",
        "TasksMax": "128",
        "CPUQuota": "250%",
    }


def test_wants_implicit_run_detection() -> None:
    assert cli._wants_implicit_run(["--cluster-config", "/tmp/cluster.yaml", "python", "train.py"])
    assert not cli._wants_implicit_run(["--cluster-config", "/tmp/cluster.yaml", "overview"])


def test_register_and_list_machines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cluster_path = tmp_path / "cluster.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lab-orch",
            "--cluster-config",
            str(cluster_path),
            "register-machine",
            "--host",
            "node33",
            "--local",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "registered machine" in out

    monkeypatch.setattr(
        sys,
        "argv",
        ["lab-orch", "--cluster-config", str(cluster_path), "machines"],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "node33" in out


def test_print_overview_probe_error(capsys) -> None:
    cli.print_overview(
        [
            NodeSnapshot(
                node_id="node34",
                ip="node34",
                hostname="node34",
                cpus_total=0.0,
                gpus_total=0.0,
                cpu_percent=100.0,
                memory_total_gb=0.0,
                memory_available_gb=0.0,
                gpu_util_avg=100.0,
                gpu_memory_free_gb=0.0,
                gpus_in_use=0.0,
                gpu_users=[],
                extras={"probe_error": "ssh: permission denied"},
            )
        ],
        {},
        {},
    )
    out = capsys.readouterr().out
    assert "probe-error" in out
    assert "permission denied" in out
    assert "n/a" in out


def test_print_overview_gpu_indices(capsys) -> None:
    cli.print_overview(
        [
            NodeSnapshot(
                node_id="node33",
                ip="10.0.0.33",
                hostname="node33",
                cpus_total=16.0,
                gpus_total=3.0,
                cpu_percent=1.0,
                memory_total_gb=64.0,
                memory_available_gb=48.0,
                gpu_util_avg=12.5,
                gpu_memory_free_gb=30.0,
                gpus_in_use=1.0,
                gpu_users=["alice"],
                extras={
                    "gpu_details": [
                        {"index": 0, "proc_count": 0.0},
                        {"index": 1, "proc_count": 0.0},
                        {"index": 2, "proc_count": 1.0},
                    ]
                },
            )
        ],
        {},
        {"node33": {1}},
    )
    out = capsys.readouterr().out
    assert "GPU idx free" in out
    assert "GPU idx resv" in out
    assert "node33" in out
    assert "0" in out
    assert "1" in out


def test_print_doctor(capsys) -> None:
    cli.print_doctor(
        [
            {
                "host": "node33",
                "mode": "ssh",
                "ssh_ok": True,
                "python_ok": True,
                "systemd_ok": True,
                "probe_ok": True,
                "probe_detail": "host=node33 ip=10.0.0.33 cpus=16 gpus=4",
            },
            {
                "host": "node34",
                "mode": "ssh",
                "ssh_ok": True,
                "python_ok": False,
                "python_detail": "python not found",
            },
        ]
    )
    out = capsys.readouterr().out
    assert "HOST" in out
    assert "SYSTEMD" in out
    assert "node33" in out
    assert "node34" in out
    assert "python not found" in out


def test_init_config_and_apply_runtime_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config_file = tmp_path / "config.yaml"
    shared_root = tmp_path / "shared"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lab-orch",
            "--config-file",
            str(config_file),
            "init-config",
            "--root",
            str(shared_root),
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "wrote config" in out

    config = load_app_config(config_file)
    assert config is not None
    assert config.cluster_config == str(shared_root / "cluster.yaml")
    assert config.db == str(shared_root / "jobs.db")
    assert config.logs_dir == str(shared_root / "logs")
    assert config.state_dir == str(shared_root / "state")

    args = argparse.Namespace(
        config_file=str(config_file),
        cluster_config=None,
        db=None,
        logs_dir=None,
        state_dir=None,
    )
    cli.apply_runtime_defaults(args)
    assert args.cluster_config == str(shared_root / "cluster.yaml")
    assert args.db == str(shared_root / "jobs.db")
    assert args.logs_dir == str(shared_root / "logs")
    assert args.state_dir == str(shared_root / "state")


def test_main_doctor_command(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class FakeOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def doctor(self):
            return [
                {
                    "host": "node33",
                    "mode": "ssh",
                    "ssh_ok": True,
                    "python_ok": True,
                    "systemd_ok": True,
                    "probe_ok": True,
                    "probe_detail": "host=node33 ip=10.0.0.33 cpus=16 gpus=4",
                }
            ]

    monkeypatch.setattr(cli, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lab-orch",
            "--cluster-config",
            "/tmp/cluster.yaml",
            "--db",
            "/tmp/jobs.db",
            "--logs-dir",
            "/tmp/logs",
            "--state-dir",
            "/tmp/state",
            "doctor",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    assert "node33" in out
    assert "PROBE" in out


def test_main_implicit_run_command(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def submit(self, request):
            captured["request"] = request
            return {
                "job_id": "job123",
                "status": "QUEUED",
                "job_mode": "single",
                "node_hostname": "bluth",
                "scheduler_name": "balanced",
                "policy_name": "static",
                "backend_name": "ssh-systemd",
                "retry_attempts": 0,
                "retry_max": 0,
                "placement_reason": "cpu_free_est=64.0, gpu_free_est=1.0",
                "allocations": [
                    {"node_hostname": "bluth", "gpus": 1.0, "cpus": 1.0}
                ],
                "log_path": "/tmp/job123.log",
            }

    monkeypatch.setattr(cli, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orch",
            "--cluster-config",
            "/tmp/cluster.yaml",
            "--db",
            "/tmp/jobs.db",
            "--logs-dir",
            "/tmp/logs",
            "--state-dir",
            "/tmp/state",
            "--host",
            "bluth",
            "--gpus",
            "1",
            "--detach",
            "python",
            "train.py",
            "--config",
            "conf.yaml",
        ],
    )
    cli.main()
    out = capsys.readouterr().out
    request = captured["request"]
    assert request.command == "python train.py --config conf.yaml"
    assert request.gpus == 1.0
    assert request.metadata["target_host"] == "bluth"
    assert "submitted job_id=job123" in out


def test_follow_logs_routes_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    stdout_path = tmp_path / "job.log"
    stderr_path = tmp_path / "job.stderr.log"
    stdout_path.write_text("out-line\n", encoding="utf-8")
    stderr_path.write_text("err-line\n", encoding="utf-8")

    class FakeOrchestrator:
        def __init__(self) -> None:
            self._calls = 0

        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            self._calls += 1
            return {"status": "SUCCEEDED" if self._calls > 1 else "RUNNING"}

        def log_sources(self, job_id: str):
            _ = job_id
            return [
                {"label": "job=abc stdout", "path": str(stdout_path), "stream": "stdout"},
                {"label": "job=abc stderr", "path": str(stderr_path), "stream": "stderr"},
            ]

    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    cli.follow_logs(FakeOrchestrator(), "abc", n_lines=100, poll_seconds=0.01)
    captured = capsys.readouterr()
    assert "out-line" in captured.out
    assert "err-line" in captured.err


def test_follow_logs_waits_for_late_stdout_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    stdout_path = tmp_path / "job.log"
    stderr_path = tmp_path / "job.stderr.log"

    class FakeOrchestrator:
        def __init__(self) -> None:
            self._calls = 0

        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            self._calls += 1
            return {"status": "SUCCEEDED"}

        def log_sources(self, job_id: str):
            _ = job_id
            return [
                {"label": "job=abc stdout", "path": str(stdout_path), "stream": "stdout"},
                {"label": "job=abc stderr", "path": str(stderr_path), "stream": "stderr"},
            ]

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            stdout_path.write_text("late-out\n", encoding="utf-8")

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    cli.follow_logs(FakeOrchestrator(), "abc", n_lines=100, poll_seconds=0.01)
    captured = capsys.readouterr()
    assert "late-out" in captured.out
