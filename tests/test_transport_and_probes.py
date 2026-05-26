from __future__ import annotations

import subprocess

import pytest

from lab_orchestrator.probes import collect_cluster_snapshots
from lab_orchestrator.registry import ClusterRegistry, MachineSpec
from lab_orchestrator.transport import SSHSystemdTransport


def test_transport_probe_and_show(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check, stdout, stderr, text, timeout):
        _ = check, stdout, stderr, text, timeout
        calls.append(list(cmd))
        if "show" in cmd[-1]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="ActiveState=active\nExecMainStatus=0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"node_id":"node33","ip":"10.0.0.33","hostname":"node33","cpus_total":8.0,"gpus_total":1.0,"cpu_percent":5.0,"memory_total_gb":32.0,"memory_available_gb":20.0,"gpu_util_avg":0.0,"gpu_memory_free_gb":10.0,"gpus_in_use":0.0,"gpu_users":[],"extras":{}}\n',
            stderr="",
        )

    monkeypatch.setattr("lab_orchestrator.transport.subprocess.run", fake_run)
    transport = SSHSystemdTransport()
    machine = MachineSpec(host="node33", local=True)
    payload = transport.probe_machine(machine)
    assert payload["hostname"] == "node33"
    status = transport.show_unit(machine, "demo.service")
    assert status["ActiveState"] == "active"
    assert calls


def test_transport_remote_ssh_does_not_wrap_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd, check, stdout, stderr, text, timeout):
        _ = check, stdout, stderr, text, timeout
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}\n", stderr="")

    monkeypatch.setattr("lab_orchestrator.transport.subprocess.run", fake_run)
    transport = SSHSystemdTransport()
    machine = MachineSpec(host="node34", ssh_user="alice", local=False)
    transport.probe_machine(machine)
    cmd = seen[0]
    assert cmd[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    assert cmd[5] == "alice@node34"
    assert "bash" not in cmd


def test_transport_start_unit_applies_default_and_custom_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd, check, stdout, stderr, text, timeout):
        _ = check, stdout, stderr, text, timeout
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("lab_orchestrator.transport.subprocess.run", fake_run)
    transport = SSHSystemdTransport()
    machine = MachineSpec(host="node33", local=True)
    transport.start_unit(
        machine=machine,
        unit_name="demo.service",
        config_path="/tmp/runner.json",
        description="demo",
        properties={"RuntimeMaxSec": "60s", "MemoryMax": "8G"},
    )

    script = seen[0][-1]
    assert "systemd-run" in script
    assert "Type=exec" in script
    assert "TasksMax=4096" in script
    assert "RuntimeMaxSec=60s" in script
    assert "MemoryMax=8G" in script


def test_collect_cluster_snapshots_with_fallback() -> None:
    class FakeTransport:
        def probe_machine(self, machine, timeout_s=8.0):
            if machine.host == "bad":
                raise RuntimeError("boom")
            return {
                "node_id": machine.host,
                "ip": "127.0.0.1",
                "hostname": machine.host,
                "cpus_total": 8.0,
                "gpus_total": 0.0,
                "cpu_percent": 1.0,
                "memory_total_gb": 16.0,
                "memory_available_gb": 8.0,
                "gpu_util_avg": 0.0,
                "gpu_memory_free_gb": 0.0,
                "gpus_in_use": 0.0,
                "gpu_users": [],
                "extras": {},
            }

    registry = ClusterRegistry(
        machines=[MachineSpec(host="good"), MachineSpec(host="bad")]
    )
    snapshots = collect_cluster_snapshots(registry, FakeTransport())
    assert len(snapshots) == 2
    fallback = next(item for item in snapshots if item.node_id == "bad")
    assert fallback.cpu_percent == 100.0
