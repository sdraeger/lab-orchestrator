# Lab Orchestrator

`lab-orchestrator` runs ordinary shell or Python jobs from one lab machine while using compute resources on another machine.

This refactor removes Ray entirely. The control plane is now:

- a local CLI and SQLite state database,
- a YAML registry of machines,
- SSH to the target host,
- `systemd-run --user` on the target host for detached job supervision,
- shared NAS paths for code, logs, and outputs.

That matches the actual problem better than a distributed Python runtime: pick a host, start a process there, track it, and stream logs back through a feel-local CLI.

## Requirements

- Linux machines with shared NAS paths.
- Passwordless SSH from the submitting machine to each registered machine.
- `systemd-run` and `systemctl` available on each target machine.
- A working user systemd manager on each target machine.
- `python3` on each target machine.
- `nvidia-smi` on GPU hosts if you want live GPU probing and automatic GPU index selection.

`lab-orchestrator` exports:

- `XDG_RUNTIME_DIR=/run/user/<uid>`
- `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/<uid>/bus`

for SSH-launched `systemd --user` commands, so non-login shells can still manage transient user units.

## What It Gives You

- Cluster-wide overview of registered machines and live load (`lab-orch overview`)
- Free and reserved GPU index visibility in the overview table
- Best-node placement for new jobs with constraints (`--cpus`, `--gpus`)
- Multi-node distributed scheduling for GPU jobs
- Detached background execution with persistent state (SQLite)
- Policy controls: per-user quotas and host allow/deny lists
- Host-pinned execution with explicit `CUDA_VISIBLE_DEVICES` masks
- Built-in command retry policy (`--max-retries`, `--retry-backoff-seconds`)
- Health checks for SSH, Python, user-systemd, and probing (`lab-orch doctor`)
- Job lifecycle commands: register, overview, submit, run, jobs, status, logs, cancel

## Install

```bash
cd /path/to/lab-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Installing the package gives you both `lab-orch` and the shorter `orch` entrypoint.

## Bootstrap Shared Config

Set one shared root once and stop exporting `LAB_ORCH_*` variables in every shell:

```bash
lab-orch init-config --root /nadata/cnl/home/$USER/.lab_orch_shared
```

This writes:

```bash
~/.lab_orch/config.yaml
```

with default paths for:

- `cluster.yaml`
- `jobs.db`
- `logs/`
- `state/`

Runtime path resolution is:

- explicit CLI flags
- `LAB_ORCH_*` environment variables
- `~/.lab_orch/config.yaml`
- built-in fallback under `~/.lab_orch/`

## Machine Registry

The cluster registry is a YAML file, by default:

```bash
~/.lab_orch/cluster.yaml
```

Example:

```yaml
ssh_user: your-username
machines:
  - host: node33
    ssh_host: node33
    python_bin: python3
  - host: node34
    ssh_host: node34
    ssh_user: your-username
    python_bin: python3
    labels:
      - gpu
```

You can edit the file directly or use the CLI:

```bash
lab-orch register-machine --host node33 --ssh-user your-username
lab-orch register-machine --host node34 --ssh-user your-username
lab-orch register-machine --host puma --ssh-user your-username --label gpu
lab-orch machines
```

`--local` exists for one-host testing, but a shared registry should normally register every machine by host and SSH user. When the current machine matches the registered host name, `lab-orch` automatically executes locally without SSH.

Remove a machine:

```bash
lab-orch unregister-machine --host puma
```

## Overview

Probe all registered machines:

```bash
lab-orch overview
```

This reports:

- CPU load
- free memory
- estimated free GPU count
- schedulable free GPU indices
- reserved GPU indices from active tracked jobs
- detected GPU users

## Doctor

Check the remote execution prerequisites on every registered machine:

```bash
lab-orch doctor
```

This validates:

- SSH connectivity
- the registered remote Python path
- `systemctl --user` availability
- full probe execution

## Policy Controls

Start from `policy.example.yaml`:

```bash
cp policy.example.yaml policy.yaml
```

Then use it globally:

```bash
lab-orch --policy-config policy.yaml overview
lab-orch --policy-config policy.yaml jobs
```

CLI flags can override file values:

```bash
lab-orch --policy-config policy.yaml --max-gpus-per-user 4 --allow-host node34 overview
```

## Submit Jobs

Submit a normal job and let the scheduler pick the host:

```bash
lab-orch submit \
  --name resnet50_trial \
  --command "python train.py --config confs/r50.yaml" \
  --cpus 4 \
  --gpus 1 \
  --scheduler fair-share \
  --max-retries 2 \
  --retry-backoff-seconds 3 \
  --metadata experiment=baseline \
  --workdir /path/to/my-project
```

Pin to one host and choose exact GPUs:

```bash
lab-orch submit \
  --name pinned_train \
  --command "python train.py" \
  --target-host node34 \
  --visible-gpus 0,2
```

Submit from YAML:

```bash
lab-orch submit --job-config job.example.yaml
```

CLI flags still override the YAML file:

```bash
lab-orch submit --job-config job.example.yaml --gpus 2
```

## Run Commands Now

Use `orch` when you want the command itself to stay front and center.

During normal non-detached runs, remote stdout is streamed to your local stdout and remote stderr is streamed to your local stderr.

Auto-schedule a CPU job:

```bash
orch python script.py --config confs/a.yaml
```

Auto-schedule onto any host with one free GPU:

```bash
orch --gpus 1 python train.py --config confs/r50.yaml
```

Pin to one host while letting the orchestrator choose a free GPU index there:

```bash
orch --host node34 --gpus 1 python train.py --config confs/r50.yaml
```

Pin exact visible GPU indices:

```bash
orch --host node34 --visible-gpus 1 python train.py --config confs/r50.yaml
```

Detached:

```bash
orch --host node34 --gpus 1 --detach python train.py
```

The older explicit form still works too:

CPU-only:

```bash
lab-orch run \
  --target-host node34 \
  -- python script.py --config confs/a.yaml
```

GPU-pinned:

```bash
lab-orch run \
  --target-host node34 \
  --gpus 1 \
  -- uv run python train.py --config confs/r50.yaml
```

Detached:

```bash
lab-orch run \
  --target-host node34 \
  --gpus 1 \
  --detach \
  -- uv run python train.py
```

## Smoke Script

The repo includes a small standalone smoke script at `test.py` for periodic orchestration checks.

CPU smoke test:

```bash
orch /home/sfdraeger/miniconda3/bin/python /nadata/cnl/home/sfdraeger/lab-orchestrator/test.py
```

GPU smoke test:

```bash
orch --gpus 1 /home/sfdraeger/miniconda3/bin/python /nadata/cnl/home/sfdraeger/lab-orchestrator/test.py --require-cuda
```

Host-pinned GPU smoke test:

```bash
orch --host bluth --visible-gpus 0 /home/sfdraeger/miniconda3/bin/python /nadata/cnl/home/sfdraeger/lab-orchestrator/test.py --require-cuda --expect-host bluth --expect-visible-gpus 0
```

The script prints a single JSON payload with hostname, Torch status, CUDA visibility, and tiny CPU/GPU tensor sums. It exits non-zero if an expected host or CUDA state is wrong.

## Distributed GPU Jobs

True single-process cross-machine "one giant CUDA box" virtualization is not what `lab-orch` provides. The feasible model is distributed multi-process launch: one worker process per GPU, across one or more machines, with the usual env vars for PyTorch/JAX-style distributed execution.

This is already supported by the current implementation.

If a request cannot fit on one machine, the orchestrator can pack GPUs across nodes.

```bash
orch --gpus 6 --distributed \
  /home/sfdraeger/miniconda3/bin/python \
  /nadata/cnl/home/sfdraeger/lab-orchestrator/distributed_smoke.py \
  --require-distributed
```

Force distributed mode explicitly:

```bash
orch --gpus 4 --distributed python train_ddp.py
```

Use explicit host/GPU bindings when you want a precise multi-host virtual GPU set:

```bash
orch \
  --gpu-bind node33:0,2 \
  --gpu-bind node34:1 \
  /home/sfdraeger/miniconda3/bin/python \
  /nadata/cnl/home/sfdraeger/lab-orchestrator/distributed_smoke.py \
  --require-distributed --require-vgpu
```

Each worker process receives:

- `CUDA_VISIBLE_DEVICES`
- `WORLD_SIZE`, `RANK`, `LOCAL_RANK`
- `NODE_RANK`, `LOCAL_WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT`
- `LAB_ORCH_VGPU_COUNT`
- `LAB_ORCH_VGPU_INDEX`
- `LAB_ORCH_VGPU_IDS`
- `LAB_ORCH_VGPU_MANIFEST_JSON`

`torch.cuda.device_count()` remains node-local. The unified cross-host resource set is exposed as a logical virtual GPU pool `0..N-1` through `lab-orch` env vars and the helper module `lab_orchestrator.vgpu`.

Inside the called script:

```python
from lab_orchestrator.vgpu import load_virtual_gpu_context

ctx = load_virtual_gpu_context()
print(ctx.ids)           # e.g. [0, 1, 2, 3]
print(ctx.current_index) # this worker's global virtual GPU id
print(ctx.current)       # host / node / physical GPU metadata
```

Your script must still support distributed launch via env vars if it wants to use compute from more than one machine. The virtual GPU helper gives you a unified logical pool; it does not make remote GPUs appear as local CUDA devices inside one process.

The repo includes `distributed_smoke.py` as a periodic multi-node smoke test. It initializes `torch.distributed` when `WORLD_SIZE > 1`, performs an `all_reduce`, and prints one JSON payload per rank including the virtual GPU context.

In `--backend auto` mode, `distributed_smoke.py` prefers `gloo` for multi-host jobs and `nccl` for single-host CUDA jobs. That makes the default smoke test validate orchestration, rendezvous, and GPU assignment without depending on cluster-specific NCCL tuning. Use `--backend nccl` when you explicitly want to validate multi-node NCCL as well.

## Job Lifecycle

```bash
lab-orch jobs --refresh
lab-orch status <job_id>
lab-orch logs <job_id> --lines 200
lab-orch logs <job_id> --follow
lab-orch logs <job_id> --all
lab-orch cancel <job_id>
```

## Notes

- Jobs are started as transient user services with `systemd-run --user`.
- Logs and runner state live on the shared filesystem, so the CLI can tail them locally.
- Automatic GPU selection uses live `nvidia-smi` probing plus reservations from tracked jobs.
- If `nvidia-smi` is unavailable, GPU jobs can still run when you pin exact GPUs with `--visible-gpus` or `--gpu-bind`.
