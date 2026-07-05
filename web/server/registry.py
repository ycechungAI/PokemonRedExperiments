"""SQLite-backed registry of training runs (history for the dashboard).

One row per run: what was launched, when, how it ended, and where its logs and
engine checkpoints live. Kept deliberately small — it records runs, it does not
manage processes (that's RunManager).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent / ".runs" / "registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    command     TEXT NOT NULL,
    params      TEXT NOT NULL,       -- JSON of the launch options
    status      TEXT NOT NULL,       -- running | stopped | exited | failed
    started_at  TEXT NOT NULL,       -- ISO8601
    ended_at    TEXT,
    returncode  INTEGER,
    log_path    TEXT,
    exp_id      TEXT                  -- engine checkpoint dir name, if discovered
);
"""


class Registry:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_run(
        self,
        run_id: str,
        command: str,
        params: dict[str, Any],
        started_at: str,
        log_path: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, command, params, status, started_at, log_path) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, command, json.dumps(params), started_at, log_path),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        ended_at: str,
        returncode: Optional[int],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ?, returncode = ? WHERE id = ?",
                (status, ended_at, returncode, run_id),
            )

    def set_exp_id(self, run_id: str, exp_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET exp_id = ? WHERE id = ?", (exp_id, run_id))

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"]) if d.get("params") else {}
        return d
