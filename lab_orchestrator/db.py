from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


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
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_node_id ON jobs(node_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alloc_node_id ON job_allocations(node_id);
"""


EXTRA_JOB_COLUMNS: dict[str, str] = {
    "job_mode": "TEXT NOT NULL DEFAULT 'single'",
    "ray_actor_names_json": "TEXT",
    "placement_json": "TEXT",
}


class JobDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            self._ensure_extra_columns(con)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _ensure_extra_columns(self, con: sqlite3.Connection) -> None:
        cols = {row[1] for row in con.execute("PRAGMA table_info(jobs)").fetchall()}
        for name, sql_type in EXTRA_JOB_COLUMNS.items():
            if name in cols:
                continue
            con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")

    def insert_job(self, row: dict[str, Any]) -> None:
        fields = list(row.keys())
        placeholders = ", ".join("?" for _ in fields)
        sql = f"INSERT INTO jobs ({', '.join(fields)}) VALUES ({placeholders})"
        with self._connect() as con:
            con.execute(sql, [row[k] for k in fields])

    def set_job_allocations(
        self, job_id: str, allocations: list[dict[str, Any]]
    ) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM job_allocations WHERE job_id = ?", (job_id,))
            for alloc in allocations:
                con.execute(
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

    def get_job_allocations(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM job_allocations WHERE job_id = ? ORDER BY node_hostname",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return
        fields = list(updates.keys())
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as con:
            con.execute(
                f"UPDATE jobs SET {set_clause} WHERE job_id = ?",
                [updates[k] for k in fields] + [job_id],
            )

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

    def list_active_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM jobs WHERE status IN ('QUEUED', 'RUNNING')"
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            out["allocations"] = self.get_job_allocations(out["job_id"])
            output.append(out)
        return output

    def resource_reservations_by_node(self) -> dict[str, dict[str, float]]:
        reservations: dict[str, dict[str, float]] = {}
        with self._connect() as con:
            alloc_rows = con.execute(
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
            legacy_rows = con.execute(
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

    @staticmethod
    def encode_env(env: dict[str, str]) -> str:
        return json.dumps(env, sort_keys=True)

    @staticmethod
    def decode_env(env_json: str) -> dict[str, str]:
        return json.loads(env_json)
