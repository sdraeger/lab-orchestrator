from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from lab_orchestrator import bootstrap, probes, ray_runtime, utils


def test_utils_parse_env_pairs_and_utc_now_iso() -> None:
    payload = utils.parse_env_pairs(["A=1", "B=hello=world", " C = spaced "])
    assert payload == {"A": "1", "B": "hello=world", "C": " spaced "}
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        utils.parse_env_pairs(["BAD"])
    with pytest.raises(ValueError, match="empty key"):
        utils.parse_env_pairs(["=x"])
    now = utils.utc_now_iso()
    assert "T" in now
    assert "+" in now or now.endswith("Z")


def test_ssh_run_dry_run_and_exec(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        calls.append(cmd)
        assert check is True

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    bootstrap._ssh_run("node1", "echo hi", ssh_user="user", dry_run=True)
    out = capsys.readouterr().out
    assert "DRY-RUN:" in out
    assert "user@node1" in out

    bootstrap._ssh_run("node1", "echo hi", ssh_user=None, dry_run=False)
    assert calls == [["ssh", "node1", "echo hi"]]


def test_ray_exec_prefix_variants() -> None:
    explicit = bootstrap._ray_exec_prefix("/opt/ray/bin/ray")
    assert "/opt/ray/bin/ray" in explicit
    assert "ray binary not found" in explicit

    autodetect = bootstrap._ray_exec_prefix(None)
    assert "command -v ray" in autodetect
    assert "ray binary not found" in autodetect


def test_bootstrap_cluster_validation_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        bootstrap.bootstrap_cluster(str(bad_cfg))

    missing_head = tmp_path / "missing_head.yaml"
    missing_head.write_text("head: ''\nworkers: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'head'"):
        bootstrap.bootstrap_cluster(str(missing_head))

    missing_head_addr = tmp_path / "missing_head_addr.yaml"
    missing_head_addr.write_text("head: node1\nhead_address: ''\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'head_address'"):
        bootstrap.bootstrap_cluster(str(missing_head_addr))

    bad_workers = tmp_path / "bad_workers.yaml"
    bad_workers.write_text("head: node1\nworkers: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workers' must be a YAML list"):
        bootstrap.bootstrap_cluster(str(bad_workers))

    good = tmp_path / "cluster.yaml"
    good.write_text(
        "\n".join(
            [
                "head: node1",
                "head_address: 10.1.1.1",
                "workers:",
                "  - node2",
                "  - ''",
                "ssh_user: user",
                "ray_port: 7000",
                "dashboard_port: 9000",
                "ray_bin: /custom/ray",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, str, str | None, bool]] = []

    def fake_ssh_run(host: str, cmd: str, ssh_user: str | None, dry_run: bool) -> None:
        calls.append((host, cmd, ssh_user, dry_run))

    monkeypatch.setattr(bootstrap, "_ssh_run", fake_ssh_run)
    bootstrap.bootstrap_cluster(str(good), dry_run=True)

    assert [c[0] for c in calls] == ["node1", "node2"]
    assert all(c[2] == "user" for c in calls)
    assert all(c[3] is True for c in calls)
    assert (
        "--head --port=7000 --dashboard-host=0.0.0.0 --dashboard-port=9000"
        in calls[0][1]
    )
    assert "--address='10.1.1.1:7000'" in calls[1][1]


def test_stop_cluster_validation_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a YAML mapping"):
        bootstrap.stop_cluster(str(bad_cfg))

    good = tmp_path / "cluster.yaml"
    good.write_text(
        "\n".join(
            [
                "head: node1",
                "workers:",
                "  - node2",
                "  - ''",
                "ssh_user: bob",
                "ray_bin: /custom/ray",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, str, str | None, bool]] = []

    def fake_ssh_run(host: str, cmd: str, ssh_user: str | None, dry_run: bool) -> None:
        calls.append((host, cmd, ssh_user, dry_run))

    monkeypatch.setattr(bootstrap, "_ssh_run", fake_ssh_run)
    bootstrap.stop_cluster(str(good), dry_run=False)
    assert [c[0] for c in calls] == ["node1", "node2"]
    assert all('"$RAY_BIN" stop --force' in c[1] for c in calls)


def test_init_ray_variants_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple] = []
    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: True)
    ray_runtime.init_ray("auto", "ns")
    assert events == []

    monkeypatch.setattr(ray_runtime.ray, "is_initialized", lambda: False)

    def fake_init(**kwargs):
        events.append(tuple(sorted(kwargs.items())))

    monkeypatch.setattr(ray_runtime.ray, "init", fake_init)
    ray_runtime.init_ray("", "ns1")
    ray_runtime.init_ray("local", "ns2")
    ray_runtime.init_ray("10.0.0.1:6379", "ns3")
    assert ("namespace", "ns1") in events[0]
    assert ("namespace", "ns2") in events[1]
    assert ("address", "10.0.0.1:6379") in events[2]

    def raise_init(**kwargs):
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(ray_runtime.ray, "init", raise_init)
    with pytest.raises(RuntimeError, match="Could not connect to Ray"):
        ray_runtime.init_ray("10.0.0.2:6379", "ns")


def test_query_nvidia_smi_and_numeric_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_run(**kwargs):
        _ = kwargs
        raise OSError("nope")

    monkeypatch.setattr(probes.subprocess, "run", raise_run)
    assert probes._query_nvidia_smi(["nvidia-smi"]) == []

    def non_zero(*args, **kwargs):
        _ = args, kwargs
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(probes.subprocess, "run", non_zero)
    assert probes._query_nvidia_smi(["nvidia-smi"]) == []

    def empty_ok(*args, **kwargs):
        _ = args, kwargs
        return types.SimpleNamespace(returncode=0, stdout="   ")

    monkeypatch.setattr(probes.subprocess, "run", empty_ok)
    assert probes._query_nvidia_smi(["nvidia-smi"]) == []

    def csv_ok(*args, **kwargs):
        _ = args, kwargs
        return types.SimpleNamespace(returncode=0, stdout="a, b \n c,d")

    monkeypatch.setattr(probes.subprocess, "run", csv_ok)
    assert probes._query_nvidia_smi(["nvidia-smi"]) == [["a", "b"], ["c", "d"]]
    assert probes.float_or_zero("1.5") == 1.5
    assert probes.float_or_zero("x") == 0.0
    assert probes.int_or_none("2") == 2
    assert probes.int_or_none("x") is None


def test_probe_local_and_collect_cluster_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_local_fn = probes._probe_local._function

    class FakeVM:
        total = 8 * 1024**3
        available = 5 * 1024**3

    class FakePsutil:
        @staticmethod
        def cpu_percent(interval: float) -> float:
            assert interval == 0.15
            return 12.5

        @staticmethod
        def virtual_memory() -> FakeVM:
            return FakeVM()

        class Process:
            def __init__(self, pid: int):
                self.pid = pid

            def username(self) -> str:
                if self.pid == 999:
                    raise RuntimeError("no user")
                return "alice"

    monkeypatch.setattr(probes.socket, "gethostname", lambda: "host-a")
    monkeypatch.setattr(probes.shutil, "which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(
        probes,
        "_query_nvidia_smi",
        lambda cmd: (
            [["0", "uuid0", "1000", "100", "20"], ["bad"]]
            if "query-gpu" in cmd[1]
            else [["uuid0", "123", "50"], ["uuid0", "999", "80"], ["bad"]]
        ),
    )

    payload = probe_local_fn("node-id", "10.0.0.1", 16.0, 2.0)
    assert payload["hostname"] == "host-a"
    assert payload["cpus_total"] == 16.0
    assert payload["gpus_total"] == 1.0
    assert payload["gpu_users"] == ["alice"]
    assert payload["gpus_in_use"] == 1.0

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(probes.shutil, "which", lambda cmd: None)
    payload_no_gpu = probe_local_fn("node-id", "10.0.0.1", 8.0, 0.0)
    assert payload_no_gpu["cpu_percent"] == 0.0
    assert payload_no_gpu["memory_total_gb"] == 0.0

    records: list[tuple[str, str, float, float]] = []

    class FakeProbeInvoker:
        def remote(self, node_id: str, node_ip: str, cpus: float, gpus: float):
            records.append((node_id, node_ip, cpus, gpus))
            return {"node_id": node_id, "ip": node_ip}

    class FakeProbeRemote:
        def options(self, **kwargs):
            _ = kwargs
            return FakeProbeInvoker()

    monkeypatch.setattr(probes, "_probe_local", FakeProbeRemote())
    monkeypatch.setattr(
        probes.ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "b" * 56,
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"CPU": 8, "GPU": 1},
            },
            {
                "Alive": True,
                "NodeID": "",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"CPU": 4, "GPU": 0},
            },
            {"Alive": False},
        ],
    )
    monkeypatch.setattr(probes.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        probes,
        "NodeSnapshot",
        lambda **kw: types.SimpleNamespace(
            hostname=f"host-{kw['ip']}", **kw, gpu_users=[], extras={}
        ),
    )
    snapshots = probes.collect_cluster_snapshots()
    assert [s.ip for s in snapshots] == ["10.0.0.1", "10.0.0.2"]
    assert len(records) == 2
