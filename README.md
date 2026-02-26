# Lab Orchestrator (Ray)

A lightweight system to run and track experiment scripts across lab machines with Ray.

## What it gives you

- Cluster-wide overview of machine load and GPU usage (`lab-orch overview`)
- Best-node placement for new jobs with constraints (`--cpus`, `--gpus`)
- Multi-node distributed scheduling for large GPU requests (e.g. `8 GPUs` across machines)
- Detached background execution with persistent state (SQLite)
- Job lifecycle commands: submit, list, status, logs, cancel
- Optional SSH bootstrap for starting/stopping Ray on a head+worker set

## Install

```bash
cd /path/to/lab-orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 1) Start your Ray cluster (optional helper)

Create `cluster.yaml`:

```yaml
head: node32
head_address: node32
workers:
  - node34
  - puma
ssh_user: your-username
ray_port: 6379
dashboard_port: 8265
ray_bin: /path/to/ray
```

`head` is used for SSH login. `head_address` is the address workers use to connect to head (set this to an IP if needed).
`ray_bin` is optional; set it when `ray` is not on remote default PATH.

Dry-run first:

```bash
lab-orch bootstrap --config cluster.yaml --dry-run
```

Then execute:

```bash
lab-orch bootstrap --config cluster.yaml
```

Stop later:

```bash
lab-orch stop-cluster --config cluster.yaml
```

If your cluster is already running, skip bootstrap.

## 2) Inspect global resources

```bash
lab-orch --cluster-config cluster.yaml --ray-address auto overview
```

When `--ray-address auto` is used with `--cluster-config`, it resolves to `head_address:ray_port`.
This probes each Ray node and reports CPU load, free memory, estimated free GPUs, and detected GPU users.

## 3) Submit experiments by resource needs

```bash
lab-orch submit \
  --name resnet50_trial \
  --command "python train.py --config confs/r50.yaml" \
  --cpus 4 \
  --gpus 1 \
  --workdir /path/to/my-project
```

The scheduler picks the best matching node and pins the job there.

If a request cannot fit on one machine (for example `--gpus 8`), the orchestrator can pack GPUs across nodes and run a single distributed job ID.

```bash
lab-orch --cluster-config cluster.yaml --ray-address auto submit \
  --name ddp_8gpu \
  --command "python train_ddp.py --config confs/big.yaml" \
  --cpus 32 \
  --gpus 8
```

Force distributed mode explicitly:

```bash
lab-orch --cluster-config cluster.yaml --ray-address auto submit \
  --name ddp_force \
  --distributed \
  --command "python train_ddp.py" \
  --cpus 16 \
  --gpus 4
```

In distributed mode, each worker process gets:

- `WORLD_SIZE`, `RANK`, `LOCAL_RANK`
- `NODE_RANK`, `LOCAL_WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT`

Your training script must support multi-process distributed execution (for example PyTorch DDP with env-based initialization).

You can also submit from YAML config:

```yaml
# job.yaml
name: resnet50_trial
command: python train.py --config confs/r50.yaml
cpus: 4
gpus: 1
distributed: false
workdir: /path/to/my-project
env:
  WANDB_MODE: offline
```

```bash
lab-orch --cluster-config cluster.yaml --ray-address auto submit --job-config job.yaml
```

CLI flags override YAML values, so this works too:

```bash
lab-orch submit --job-config job.yaml --gpus 2
```

## 4) Track and manage jobs

```bash
lab-orch --cluster-config cluster.yaml --ray-address auto jobs --refresh
lab-orch --cluster-config cluster.yaml --ray-address auto status <job_id>
lab-orch --cluster-config cluster.yaml --ray-address auto logs <job_id> --lines 200
lab-orch --cluster-config cluster.yaml --ray-address auto logs <job_id> --follow
lab-orch --cluster-config cluster.yaml --ray-address auto logs <job_id> --all
lab-orch --cluster-config cluster.yaml --ray-address auto cancel <job_id>
```

`logs` merges stdout+stderr (jobs are launched with stderr redirected into stdout).
For `FAILED` jobs, `logs` prints full history by default.
Use `--follow` to stream while the job is alive.

## Defaults and paths

- DB: `~/.lab_orch/jobs.db`
- Logs: `~/.lab_orch/logs/<job_id>.log`
- Namespace: `lab-orchestrator`
- Ray address: `auto`
- Cluster config auto-discovery: `./cluster.yaml`, `./cluster.example.yaml`, `~/.lab_orch/cluster.yaml`

Override via flags or env vars:

- `LAB_ORCH_RAY_ADDRESS`
- `LAB_ORCH_CLUSTER_CONFIG`
- `LAB_ORCH_NAMESPACE`
- `LAB_ORCH_DB`
- `LAB_ORCH_LOGS`

## Notes

- This scheduler is conservative: it combines Ray reservations from tracked jobs with live node load/GPU process probes.
- GPU usage from non-Ray processes is detected via `nvidia-smi` when available.
- If `nvidia-smi` is absent on a node, GPU fields fall back to Ray-advertised GPU count.
- Distributed jobs reserve resources per node in the DB, so `overview` and future placements account for multi-node reservations.
