from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pytest

from lab_orchestrator import cli
from lab_orchestrator.models import NodeSnapshot
from lab_orchestrator.policy import StaticSubmissionPolicy


class _DummyOrch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._status_rows: list[dict] = []

    def overview(self):
        return {"nodes": [], "reservations": {}}

    def submit(self, request):
        _ = request
        return {
            "job_id": "j1",
            "status": "RUNNING",
            "job_mode": "single",
            "node_hostname": "node1",
            "placement_reason": "fit",
            "allocations": [{"node_hostname": "node1", "gpus": 1.0, "cpus": 2.0}],
            "log_path": "/tmp/j1.log",
            "scheduler_name": "balanced",
            "policy_name": "static",
            "retry_attempts": 0,
            "retry_max": 1,
        }

    def list_jobs(self, limit: int = 50, refresh: bool = False):
        _ = limit, refresh
        return []

    def status(self, job_id: str, refresh: bool = True):
        _ = job_id, refresh
        if self._status_rows:
            return self._status_rows.pop(0)
        return {"status": "RUNNING", "job_id": "j1"}

    def cancel(self, job_id: str, grace_seconds: int = 20):
        _ = job_id, grace_seconds
        return {"job_id": "j1", "status": "CANCELLED", "return_code": 137}

    def logs(self, job_id: str, n_lines: int = 100):
        _ = job_id, n_lines
        return "TAIL\n"

    def logs_all(self, job_id: str):
        _ = job_id
        return "ALL\n"

    def log_sources(self, job_id: str):
        _ = job_id
        return []


def _node() -> NodeSnapshot:
    return NodeSnapshot(
        node_id="n1",
        ip="10.0.0.1",
        hostname="host1",
        cpus_total=8.0,
        gpus_total=1.0,
        cpu_percent=12.0,
        memory_total_gb=32.0,
        memory_available_gb=20.0,
        gpu_util_avg=0.0,
        gpu_memory_free_gb=8.0,
        gpus_in_use=0.0,
    )


def test_load_config_helpers_and_policy_builder(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text("name: x\n", encoding="utf-8")
    assert cli.load_job_config(str(job_cfg))["name"] == "x"
    job_cfg.write_text("null\n", encoding="utf-8")
    assert cli.load_job_config(str(job_cfg)) == {}
    job_cfg.write_text("- x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="job config must be a YAML mapping"):
        cli.load_job_config(str(job_cfg))

    cluster_cfg = tmp_path / "cluster.yaml"
    cluster_cfg.write_text("head: node1\n", encoding="utf-8")
    assert cli.load_cluster_config(str(cluster_cfg))["head"] == "node1"
    cluster_cfg.write_text("null\n", encoding="utf-8")
    assert cli.load_cluster_config(str(cluster_cfg)) == {}
    cluster_cfg.write_text("- x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cluster config must be a YAML mapping"):
        cli.load_cluster_config(str(cluster_cfg))

    policy_cfg = tmp_path / "policy.yaml"
    policy_cfg.write_text("name: p\nmax_gpus_per_user: 2\n", encoding="utf-8")
    args = argparse.Namespace(
        policy_config=str(policy_cfg),
        max_active_jobs_per_user=3,
        max_cpus_per_user=5.5,
        max_gpus_per_user=4.0,
        allow_host=[" Host1 ", ""],
        deny_host=["Host2"],
    )
    policy = cli.build_submission_policy(args)
    assert isinstance(policy, StaticSubmissionPolicy)
    assert policy.max_active_jobs_per_user == 3
    assert policy.max_cpus_per_user == 5.5
    assert policy.max_gpus_per_user == 4.0
    assert policy.allowed_hosts == {"host1"}
    assert policy.denied_hosts == {"host2"}


def test_address_resolution_and_cluster_path_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.resolve_ray_address("10.0.0.1:6379", None) == "10.0.0.1:6379"
    assert cli._pick_cluster_config_path(str(tmp_path / "missing.yaml")) is None
    assert cli.resolve_ray_address("auto", str(tmp_path / "missing.yaml")) == "auto"

    cfg = tmp_path / "cluster.yaml"
    cfg.write_text("head_address: node-a\nray_port: 7000\n", encoding="utf-8")
    assert cli.resolve_ray_address("auto", str(cfg)) == "node-a:7000"

    cfg.write_text("head_address: ''\n", encoding="utf-8")
    assert cli.resolve_ray_address("auto", str(cfg)) == "auto"

    monkeypatch.chdir(tmp_path)
    cfg.write_text("head_address: node-a\nray_port: 7000\n", encoding="utf-8")
    assert cli._pick_cluster_config_path(None) == str(cfg)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(cli.Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    assert cli._pick_cluster_config_path(None) is None


def test_print_helpers_and_tail_helpers(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.print_overview([], {})
    assert "No live Ray nodes found." in capsys.readouterr().out

    cli.print_overview([_node()], {"n1": {"cpus": 0.0, "gpus": 0.0}})
    out = capsys.readouterr().out
    assert "HOST" in out and "host1" in out

    cli.print_jobs([])
    assert "No jobs in DB." in capsys.readouterr().out

    cli.print_jobs(
        [
            {
                "job_id": "j1",
                "status": "RUNNING",
                "job_mode": "single",
                "scheduler_name": "balanced",
                "name": "job-name",
                "node_hostname": "host1",
                "requested_cpus": 2.0,
                "requested_gpus": 1.0,
                "retry_attempts": 0,
                "retry_max": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    assert "job-name" in capsys.readouterr().out

    row = {
        "job_id": "j1",
        "name": "n",
        "status": "RUNNING",
        "job_mode": "single",
        "scheduler_name": "balanced",
        "policy_name": "static",
        "submit_user": "alice",
        "node_hostname": "host1",
        "node_ip": "10.0.0.1",
        "requested_cpus": 1.0,
        "requested_gpus": 0.0,
        "retry_max": 1,
        "retry_backoff_seconds": 0.1,
        "retry_attempts": 0,
        "created_at": "x",
        "started_at": None,
        "ended_at": None,
        "return_code": None,
        "command": "echo",
        "log_path": "/tmp/x.log",
        "error_text": None,
        "metadata_json": '{"a":1}',
        "allocations": [
            {"node_hostname": "host1", "node_ip": "10.0.0.1", "cpus": 1.0, "gpus": 0.0}
        ],
    }
    cli.print_job_status(row)
    out = capsys.readouterr().out
    assert "metadata:" in out
    assert "allocations:" in out

    monkeypatch.setattr(
        cli.yaml,
        "safe_dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    cli.print_job_status(row)
    assert "metadata_json" in capsys.readouterr().out

    p = tmp_path / "x.log"
    assert cli._read_tail_from_file(p, 3) == ""
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert cli._read_tail_from_file(p, 2) == "b\nc\n"
    rendered = cli._render_sources_from_files(
        [{"label": "L", "path": str(p)}],
        n_lines=1,
    )
    assert rendered.endswith("c\n")
    p2 = tmp_path / "y.log"
    p2.write_text("no-newline", encoding="utf-8")
    rendered_multi = cli._render_sources_from_files(
        [
            {"label": "A", "path": str(p2)},
            {"label": "B", "path": str(tmp_path / "missing.log")},
        ],
        n_lines=3,
    )
    assert rendered_multi.startswith("===== A =====")
    assert rendered_multi.endswith("\n")


def test_follow_logs_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    p = tmp_path / "job.log"
    p.write_text("start\n", encoding="utf-8")

    class FailedOrch:
        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            return {"status": "FAILED"}

        def logs_all(self, job_id: str):
            _ = job_id
            return "ALL\n"

    cli.follow_logs(FailedOrch(), "j1", n_lines=5, poll_seconds=0.1)
    assert capsys.readouterr().out == "ALL\n"

    class StreamOrch:
        def __init__(self):
            self.calls = 0

        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            self.calls += 1
            if self.calls < 2:
                return {"status": "RUNNING"}
            if self.calls == 2:
                return {"status": "FAILED"}
            return {"status": "FAILED"}

        def log_sources(self, job_id: str):
            _ = job_id
            return [{"label": "L1", "path": str(p)}]

        def logs_all(self, job_id: str):
            _ = job_id
            return "FULL\n"

    sleep_calls = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_calls["n"] += 1
        if abs(seconds - 0.25) < 1e-9 and sleep_calls["n"] == 2:
            p.write_text("start\nlate\n", encoding="utf-8")

    monkeypatch.setattr(cli.time, "sleep", fake_sleep)
    cli.follow_logs(StreamOrch(), "j1", n_lines=10, poll_seconds=0.0)
    out = capsys.readouterr().out
    assert "start" in out
    assert "FULL" in out

    p_multi = tmp_path / "multi.log"
    p_multi.write_text("x\n", encoding="utf-8")

    class ComplexOrch:
        def __init__(self):
            self.calls = 0

        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            self.calls += 1
            return {"status": "SUCCEEDED"}

        def log_sources(self, job_id: str):
            _ = job_id
            return [
                {"label": "L1", "path": str(p_multi)},
                {"label": "L2", "path": str(tmp_path / "missing.log")},
            ]

        def logs_all(self, job_id: str):
            _ = job_id
            return "NA\n"

    sleep_state = {"n": 0}

    def sleep_and_mutate(seconds: float) -> None:
        sleep_state["n"] += 1
        if sleep_state["n"] == 1:
            # Trigger size < previous branch.
            p_multi.write_text("", encoding="utf-8")
        elif abs(seconds - 0.25) < 1e-9 and sleep_state["n"] == 2:
            # Trigger late_new branch in drain loop.
            p_multi.write_text("late\n", encoding="utf-8")

    monkeypatch.setattr(cli.time, "sleep", sleep_and_mutate)
    cli.follow_logs(ComplexOrch(), "j2", n_lines=10, poll_seconds=0.0)
    out = capsys.readouterr().out
    assert "===== L1 =====" in out
    assert "te" in out

    p_outer = tmp_path / "outer.log"
    p_outer.write_text("", encoding="utf-8")

    class OuterChunkOrch:
        def __init__(self):
            self.calls = 0

        def status(self, job_id: str, refresh: bool = True):
            _ = job_id, refresh
            self.calls += 1
            return {"status": "SUCCEEDED"}

        def log_sources(self, job_id: str):
            _ = job_id
            return [
                {"label": "A", "path": str(p_outer)},
                {"label": "B", "path": str(tmp_path / "missing2.log")},
            ]

        def logs_all(self, job_id: str):
            _ = job_id
            return "X\n"

    sleep_outer = {"n": 0}

    def sleep_outer_fn(seconds: float) -> None:
        sleep_outer["n"] += 1
        if sleep_outer["n"] == 1:
            p_outer.write_text("outer\n", encoding="utf-8")

    monkeypatch.setattr(cli.time, "sleep", sleep_outer_fn)
    cli.follow_logs(OuterChunkOrch(), "j3", n_lines=10, poll_seconds=0.0)
    out = capsys.readouterr().out
    assert "===== A =====" in out
    assert "outer" in out


def test_main_command_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    shared = _DummyOrch()

    def orch_ctor(**kwargs):
        shared.kwargs = kwargs
        return shared

    monkeypatch.setattr(cli, "Orchestrator", orch_ctor)
    monkeypatch.setattr(
        cli, "resolve_placement_strategy", lambda name: type("S", (), {"name": name})()
    )
    monkeypatch.setattr(
        cli,
        "build_submission_policy",
        lambda args: StaticSubmissionPolicy(name="static"),
    )
    monkeypatch.setattr(cli, "resolve_ray_address", lambda ra, cc: "resolved")
    monkeypatch.setattr(cli, "bootstrap_cluster", lambda config, dry_run=False: None)
    monkeypatch.setattr(cli, "stop_cluster", lambda config, dry_run=False: None)
    monkeypatch.setattr(
        cli, "follow_logs", lambda orch, job_id, n_lines, poll_seconds: print("FOLLOW")
    )

    sys.argv = ["lab-orch"]
    cli.main()
    assert "usage:" in capsys.readouterr().out

    sys.argv = ["lab-orch", "bootstrap", "--config", "c.yaml", "--dry-run"]
    cli.main()
    assert "Bootstrap complete." in capsys.readouterr().out

    sys.argv = ["lab-orch", "stop-cluster", "--config", "c.yaml"]
    cli.main()
    assert "Cluster stop complete." in capsys.readouterr().out

    sys.argv = ["lab-orch", "overview"]
    cli.main()
    assert "No live Ray nodes found." in capsys.readouterr().out

    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text(
        "\n".join(
            [
                "name: cfg-name",
                "command: echo hi",
                "cpus: 3",
                "gpus: 1",
                "workdir: .",
                "distributed: false",
                "submit_user: bob",
                "max_retries: 1",
                "retry_backoff_seconds: 0.5",
                "env: {X: '1'}",
                "metadata: {k: v}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sys.argv = [
        "lab-orch",
        "submit",
        "--job-config",
        str(job_cfg),
        "--env",
        "Y=2",
        "--metadata",
        "z=3",
    ]
    cli.main()
    out = capsys.readouterr().out
    assert "submitted job_id=j1" in out
    assert "allocations:" in out
    assert "scheduler=balanced" in out

    null_cfg = tmp_path / "null_cfg.yaml"
    null_cfg.write_text(
        "name: n\ncommand: echo\nenv: null\nmetadata: null\n", encoding="utf-8"
    )
    sys.argv = ["lab-orch", "submit", "--job-config", str(null_cfg)]
    cli.main()
    assert "submitted job_id=j1" in capsys.readouterr().out

    sys.argv = ["lab-orch", "jobs", "--limit", "5", "--refresh"]
    cli.main()
    assert "No jobs in DB." in capsys.readouterr().out

    shared._status_rows = [{"job_id": "j1", "status": "RUNNING"}]
    sys.argv = ["lab-orch", "status", "j1", "--no-refresh"]
    cli.main()
    assert "job_id: j1" in capsys.readouterr().out

    sys.argv = ["lab-orch", "cancel", "j1", "--grace-seconds", "1"]
    cli.main()
    assert "status=CANCELLED" in capsys.readouterr().out

    shared._status_rows = [{"status": "FAILED"}]
    sys.argv = ["lab-orch", "logs", "j1", "--lines", "10"]
    cli.main()
    assert "ALL" in capsys.readouterr().out

    shared._status_rows = [{"status": "RUNNING"}]
    sys.argv = ["lab-orch", "logs", "j1", "--all"]
    cli.main()
    assert "ALL" in capsys.readouterr().out

    shared._status_rows = [{"status": "RUNNING"}]
    sys.argv = ["lab-orch", "logs", "j1"]
    cli.main()
    assert "TAIL" in capsys.readouterr().out

    sys.argv = ["lab-orch", "logs", "j1", "--follow"]
    cli.main()
    assert "FOLLOW" in capsys.readouterr().out


def test_main_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "resolve_placement_strategy",
        lambda name: (_ for _ in ()).throw(ValueError("bad strategy")),
    )
    sys.argv = ["lab-orch", "jobs"]
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "ERROR: bad strategy" in capsys.readouterr().err

    monkeypatch.setattr(
        cli, "resolve_placement_strategy", lambda name: type("S", (), {"name": name})()
    )
    monkeypatch.setattr(
        cli,
        "build_submission_policy",
        lambda args: StaticSubmissionPolicy(name="static"),
    )
    monkeypatch.setattr(cli, "Orchestrator", lambda **kwargs: _DummyOrch(**kwargs))
    monkeypatch.setattr(cli, "resolve_ray_address", lambda ra, cc: "resolved")

    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("name: x\nenv: []\ncommand: echo\n", encoding="utf-8")
    sys.argv = ["lab-orch", "submit", "--job-config", str(bad_cfg)]
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2

    bad_meta = tmp_path / "bad_meta.yaml"
    bad_meta.write_text("name: x\ncommand: echo\nmetadata: []\n", encoding="utf-8")
    sys.argv = ["lab-orch", "submit", "--job-config", str(bad_meta)]
    with pytest.raises(SystemExit):
        cli.main()

    sys.argv = ["lab-orch", "submit", "--command", "echo"]
    with pytest.raises(SystemExit):
        cli.main()

    sys.argv = ["lab-orch", "submit", "--name", "x"]
    with pytest.raises(SystemExit):
        cli.main()

    sys.argv = [
        "lab-orch",
        "submit",
        "--name",
        "x",
        "--command",
        "echo",
        "--max-retries",
        "-1",
    ]
    with pytest.raises(SystemExit):
        cli.main()

    sys.argv = [
        "lab-orch",
        "submit",
        "--name",
        "x",
        "--command",
        "echo",
        "--retry-backoff-seconds",
        "-0.1",
    ]
    with pytest.raises(SystemExit):
        cli.main()

    parser = cli.build_parser()
    args = parser.parse_args(["jobs"])
    assert args.cmd == "jobs"

    class _FakeParser:
        def parse_args(self):
            return argparse.Namespace(
                cmd="weird",
                ray_address="auto",
                cluster_config=None,
                namespace="ns",
                db="/tmp/db.sqlite",
                logs_dir="/tmp/logs",
                scheduler="balanced",
                policy_config=None,
                max_active_jobs_per_user=None,
                max_cpus_per_user=None,
                max_gpus_per_user=None,
                allow_host=[],
                deny_host=[],
            )

        def print_help(self):
            print("help")

    monkeypatch.setattr(cli, "build_parser", lambda: _FakeParser())
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_cli_module_entrypoint(capsys) -> None:
    sys.argv = ["lab-orch"]
    runpy.run_module("lab_orchestrator.cli", run_name="__main__")
    assert "usage:" in capsys.readouterr().out
