from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    requested_cpus REAL NOT NULL,
    requested_gpus REAL NOT NULL,
    workdir TEXT NOT NULL,
    env_json TEXT NOT NULL,
    status TEXT NOT NULL,
    submit_user TEXT NOT NULL,
    ray_actor_name TEXT NOT NULL,
    ray_namespace TEXT NOT NULL,
    node_id TEXT,
    node_ip TEXT,
    node_hostname TEXT,
    log_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    return_code INTEGER,
    error_text TEXT
);
CREATE TABLE IF NOT EXISTS job_allocations (
    job_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_ip TEXT,
    node_hostname TEXT,
    cpus REAL NOT NULL,
    gpus REAL NOT NULL,
    PRIMARY KEY (job_id, node_id)
);
CREATE TABLE IF NOT EXISTS job_gpu_leases (
    job_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_hostname TEXT,
    gpu_index INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (node_id, gpu_index),
    UNIQUE (job_id, node_id, gpu_index)
);
CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_node_id ON jobs(node_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_submit_user_status ON jobs(submit_user, status);
CREATE INDEX IF NOT EXISTS idx_alloc_node_id ON job_allocations(node_id);
CREATE INDEX IF NOT EXISTS idx_gpu_leases_job_id ON job_gpu_leases(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id_created_at ON job_events(job_id, created_at);
"""


EXTRA_JOB_COLUMNS: dict[str, str] = {
    "job_mode": "TEXT NOT NULL DEFAULT 'single'",
    "ray_actor_names_json": "TEXT",
    "placement_json": "TEXT",
    "metadata_json": "TEXT",
    "scheduler_name": "TEXT",
    "policy_name": "TEXT",
    "retry_max": "INTEGER NOT NULL DEFAULT 0",
    "retry_backoff_seconds": "REAL NOT NULL DEFAULT 5.0",
    "retry_attempts": "INTEGER NOT NULL DEFAULT 0",
    "backend_name": "TEXT NOT NULL DEFAULT 'ssh-systemd'",
    "remote_unit_name": "TEXT",
    "remote_handles_json": "TEXT",
    "state_path": "TEXT",
    "spool_path": "TEXT",
}


class JobDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            self._ensure_extra_columns(con)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 30000")
        return con

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _ensure_extra_columns(self, con: sqlite3.Connection) -> None:
        cols = {row[1] for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
        for name, sql_type in EXTRA_JOB_COLUMNS.items():
            if name in cols:
                continue
            con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")

    def insert_job(
        self, row: dict[str, Any], con: sqlite3.Connection | None = None
    ) -> None:
        fields = list(row.keys())
        placeholders = ", ".join("?" for _ in fields)
        sql = f"INSERT INTO jobs ({', '.join(fields)}) VALUES ({placeholders})"
        if con is not None:
            con.execute(sql, [row[k] for k in fields])
            return
        with self._connect() as owned:
            owned.execute(sql, [row[k] for k in fields])

    def set_job_allocations(
        self,
        job_id: str,
        allocations: list[dict[str, Any]],
        con: sqlite3.Connection | None = None,
    ) -> None:
        def _set(active: sqlite3.Connection) -> None:
            active.execute("DELETE FROM job_allocations WHERE job_id = ?", (job_id,))
            for alloc in allocations:
                active.execute(
                    """
                    INSERT INTO job_allocations (job_id, node_id, node_ip, node_hostname, cpus, gpus)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        alloc["node_id"],
                        alloc.get("node_ip"),
                        alloc.get("node_hostname"),
                        float(alloc.get("cpus", 0.0)),
                        float(alloc.get("gpus", 0.0)),
                    ),
                )

        if con is not None:
            _set(con)
            return
        with self._connect() as owned:
            _set(owned)

    def set_gpu_leases(
        self,
        job_id: str,
        leases: list[dict[str, Any]],
        con: sqlite3.Connection | None = None,
    ) -> None:
        def _set(active: sqlite3.Connection) -> None:
            active.execute("DELETE FROM job_gpu_leases WHERE job_id = ?", (job_id,))
            for lease in leases:
                try:
                    active.execute(
                        """
                        INSERT INTO job_gpu_leases
                            (job_id, node_id, node_hostname, gpu_index, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            job_id,
                            lease["node_id"],
                            lease.get("node_hostname"),
                            int(lease["gpu_index"]),
                            lease["created_at"],
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(
                        "GPU lease conflict: "
                        f"{lease.get('node_hostname') or lease['node_id']}:{int(lease['gpu_index'])} "
                        "is already reserved by another active job."
                    ) from exc

        if con is not None:
            _set(con)
            return
        with self._connect() as owned:
            _set(owned)

    def release_gpu_leases(
        self, job_id: str, con: sqlite3.Connection | None = None
    ) -> None:
        if con is not None:
            con.execute("DELETE FROM job_gpu_leases WHERE job_id = ?", (job_id,))
            return
        with self._connect() as owned:
            owned.execute("DELETE FROM job_gpu_leases WHERE job_id = ?", (job_id,))

    def active_gpu_leases_by_node(
        self, con: sqlite3.Connection | None = None
    ) -> dict[str, set[int]]:
        def _read(active: sqlite3.Connection) -> dict[str, set[int]]:
            rows = active.execute(
                """
                SELECT l.node_id, l.gpu_index
                FROM job_gpu_leases l
                JOIN jobs j ON l.job_id = j.job_id
                WHERE j.status IN ('QUEUED', 'RUNNING')
                """
            ).fetchall()
            out: dict[str, set[int]] = {}
            for row in rows:
                node_id = str(row["node_id"] or "").strip()
                if node_id:
                    out.setdefault(node_id, set()).add(int(row["gpu_index"]))
            return out

        if con is not None:
            return _read(con)
        with self._connect() as owned:
            return _read(owned)

    def insert_job_event(
        self,
        job_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        con: sqlite3.Connection | None = None,
    ) -> None:
        from .utils import utc_now_iso

        payload_json = json.dumps(payload or {}, sort_keys=True)
        values = (job_id, event_type, payload_json, utc_now_iso())
        sql = """
            INSERT INTO job_events (job_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
        """
        if con is not None:
            con.execute(sql, values)
            return
        with self._connect() as owned:
            owned.execute(sql, values)

    def list_job_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM job_events
                WHERE job_id = ?
                ORDER BY event_id ASC
                """,
                (job_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(str(item.get("payload_json") or "{}"))
            except Exception:
                item["payload"] = {}
            out.append(item)
        return out

    def get_job_allocations(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM job_allocations WHERE job_id = ? ORDER BY node_hostname",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_job(
        self, job_id: str, con: sqlite3.Connection | None = None, **updates: Any
    ) -> None:
        if not updates:
            return
        fields = list(updates.keys())
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = [updates[k] for k in fields] + [job_id]
        if con is not None:
            con.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
            return
        with self._connect() as owned:
            owned.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        out = dict(row)
        out["allocations"] = self.get_job_allocations(job_id)
        return out

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM jobs ORDER BY datetime(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            out["allocations"] = self.get_job_allocations(out["job_id"])
            output.append(out)
        return output

    def list_active_jobs(
        self, con: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        def _read(active: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = active.execute(
                "SELECT * FROM jobs WHERE status IN ('QUEUED', 'RUNNING')"
            ).fetchall()
            output: list[dict[str, Any]] = []
            for row in rows:
                out = dict(row)
                out["allocations"] = self.get_job_allocations(out["job_id"])
                output.append(out)
            return output

        if con is not None:
            return _read(con)
        with self._connect() as owned:
            return _read(owned)

    def resource_reservations_by_node(
        self, con: sqlite3.Connection | None = None
    ) -> dict[str, dict[str, float]]:
        reservations: dict[str, dict[str, float]] = {}

        def _read(active: sqlite3.Connection) -> dict[str, dict[str, float]]:
            alloc_rows = active.execute(
                """
                SELECT a.node_id, SUM(a.cpus) AS cpus, SUM(a.gpus) AS gpus
                FROM job_allocations a
                JOIN jobs j ON a.job_id = j.job_id
                WHERE j.status IN ('QUEUED', 'RUNNING')
                GROUP BY a.node_id
                """
            ).fetchall()
            for row in alloc_rows:
                reservations[row["node_id"]] = {
                    "cpus": float(row["cpus"] or 0.0),
                    "gpus": float(row["gpus"] or 0.0),
                }

            # Backward compatibility for jobs created before job_allocations existed.
            legacy_rows = active.execute(
                """
                SELECT j.node_id, SUM(j.requested_cpus) AS cpus, SUM(j.requested_gpus) AS gpus
                FROM jobs j
                WHERE j.status IN ('QUEUED', 'RUNNING')
                  AND j.node_id IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM job_allocations a WHERE a.job_id = j.job_id)
                GROUP BY j.node_id
                """
            ).fetchall()
            for row in legacy_rows:
                node_id = row["node_id"]
                if not node_id:
                    continue
                current = reservations.setdefault(node_id, {"cpus": 0.0, "gpus": 0.0})
                current["cpus"] += float(row["cpus"] or 0.0)
                current["gpus"] += float(row["gpus"] or 0.0)

            return reservations

        if con is not None:
            return _read(con)
        with self._connect() as owned:
            return _read(owned)

    def active_usage_for_user(
        self, submit_user: str, con: sqlite3.Connection | None = None
    ) -> dict[str, float]:
        def _read(active: sqlite3.Connection) -> sqlite3.Row | None:
            return active.execute(
                """
                SELECT
                    COUNT(*) AS jobs,
                    COALESCE(SUM(requested_cpus), 0.0) AS cpus,
                    COALESCE(SUM(requested_gpus), 0.0) AS gpus
                FROM jobs
                WHERE status IN ('QUEUED', 'RUNNING')
                  AND submit_user = ?
                """,
                (submit_user,),
            ).fetchone()

        row = _read(con) if con is not None else None
        if con is None:
            with self._connect() as owned:
                row = _read(owned)
        if row is None:
            return {"jobs": 0.0, "cpus": 0.0, "gpus": 0.0}
        return {
            "jobs": float(row["jobs"] or 0.0),
            "cpus": float(row["cpus"] or 0.0),
            "gpus": float(row["gpus"] or 0.0),
        }

    def active_usage_by_user(
        self, con: sqlite3.Connection | None = None
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}

        def _read(active: sqlite3.Connection) -> list[sqlite3.Row]:
            return active.execute(
                """
                SELECT
                    submit_user,
                    COUNT(*) AS jobs,
                    COALESCE(SUM(requested_cpus), 0.0) AS cpus,
                    COALESCE(SUM(requested_gpus), 0.0) AS gpus
                FROM jobs
                WHERE status IN ('QUEUED', 'RUNNING')
                GROUP BY submit_user
                """
            ).fetchall()

        rows = _read(con) if con is not None else None
        if con is None:
            with self._connect() as owned:
                rows = _read(owned)
        for row in rows:
            user = str(row["submit_user"] or "").strip()
            if not user:
                continue
            out[user] = {
                "jobs": float(row["jobs"] or 0.0),
                "cpus": float(row["cpus"] or 0.0),
                "gpus": float(row["gpus"] or 0.0),
            }
        return out

    def active_user_reservations_by_node(
        self, con: sqlite3.Connection | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        # node_id -> submit_user -> {cpus, gpus}
        out: dict[str, dict[str, dict[str, float]]] = {}

        def _read(active: sqlite3.Connection) -> None:
            rows = active.execute(
                """
                SELECT
                    a.node_id AS node_id,
                    j.submit_user AS submit_user,
                    COALESCE(SUM(a.cpus), 0.0) AS cpus,
                    COALESCE(SUM(a.gpus), 0.0) AS gpus
                FROM job_allocations a
                JOIN jobs j ON a.job_id = j.job_id
                WHERE j.status IN ('QUEUED', 'RUNNING')
                GROUP BY a.node_id, j.submit_user
                """
            ).fetchall()
            for row in rows:
                node_id = str(row["node_id"] or "").strip()
                user = str(row["submit_user"] or "").strip()
                if not node_id or not user:
                    continue
                by_user = out.setdefault(node_id, {})
                by_user[user] = {
                    "cpus": float(row["cpus"] or 0.0),
                    "gpus": float(row["gpus"] or 0.0),
                }

            legacy_rows = active.execute(
                """
                SELECT
                    j.node_id AS node_id,
                    j.submit_user AS submit_user,
                    COALESCE(SUM(j.requested_cpus), 0.0) AS cpus,
                    COALESCE(SUM(j.requested_gpus), 0.0) AS gpus
                FROM jobs j
                WHERE j.status IN ('QUEUED', 'RUNNING')
                  AND j.node_id IS NOT NULL
                  AND j.node_id != ''
                  AND NOT EXISTS (SELECT 1 FROM job_allocations a WHERE a.job_id = j.job_id)
                GROUP BY j.node_id, j.submit_user
                """
            ).fetchall()
            for row in legacy_rows:
                node_id = str(row["node_id"] or "").strip()
                user = str(row["submit_user"] or "").strip()
                if not node_id or not user:
                    continue
                by_user = out.setdefault(node_id, {})
                usage = by_user.setdefault(user, {"cpus": 0.0, "gpus": 0.0})
                usage["cpus"] += float(row["cpus"] or 0.0)
                usage["gpus"] += float(row["gpus"] or 0.0)

        if con is not None:
            _read(con)
        else:
            with self._connect() as owned:
                _read(owned)
        return out

    @staticmethod
    def encode_env(env: dict[str, str], redact: bool = False) -> str:
        payload = dict(env)
        if redact:
            payload = {
                key: ("<redacted>" if _looks_secret_key(key) else value)
                for key, value in payload.items()
            }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def decode_env(env_json: str) -> dict[str, str]:
        return json.loads(env_json)


def _looks_secret_key(key: str) -> bool:
    normalized = str(key).upper()
    secret_markers = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "API_KEY",
        "ACCESS_KEY",
        "PRIVATE_KEY",
        "CREDENTIAL",
    )
    return any(marker in normalized for marker in secret_markers)
