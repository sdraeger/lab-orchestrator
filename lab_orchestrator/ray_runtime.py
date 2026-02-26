from __future__ import annotations

import ray


def init_ray(address: str, namespace: str) -> None:
    if ray.is_initialized():
        return
    try:
        if address in {"", "local"}:
            ray.init(namespace=namespace, log_to_driver=False)
        else:
            ray.init(address=address, namespace=namespace, log_to_driver=False)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ray. "
            f"address={address!r}. "
            "If you use a remote cluster, start Ray on the head/worker nodes first "
            "(or run `lab-orch bootstrap --config <file>`)."
        ) from exc
