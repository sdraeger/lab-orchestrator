from __future__ import annotations

from lab_orchestrator.models import JobRequest
from lab_orchestrator.provenance import collect_submission_metadata


def test_collect_submission_metadata_contains_expected_fields() -> None:
    request = JobRequest(
        name="demo",
        command="python train.py",
        cpus=4.0,
        gpus=1.0,
        workdir=".",
        env={"B": "2", "A": "1"},
        distributed=True,
        max_retries=2,
        retry_backoff_seconds=1.5,
        metadata={"run_group": "ablation"},
    )
    payload = collect_submission_metadata(
        request=request,
        scheduler_name="fair-share",
        policy_name="lab-default",
        submit_user="alice",
    )
    assert payload["scheduler"] == "fair-share"
    assert payload["policy"] == "lab-default"
    assert payload["submit_user"] == "alice"
    assert payload["env_keys"] == ["A", "B"]
    assert payload["retry"]["max_retries"] == 2
    assert payload["retry"]["backoff_seconds"] == 1.5
    assert payload["user_metadata"] == {"run_group": "ablation"}
    assert isinstance(payload["request_hash"], str)
    assert len(payload["request_hash"]) == 64
