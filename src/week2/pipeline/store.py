import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from .models import Answer
from .settings import RunSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       REAL    NOT NULL,
    elapsed_seconds  REAL    NOT NULL,
    n_questions      INTEGER NOT NULL,
    n_successed      INTEGER NOT NULL,
    n_retries_total  INTEGER NOT NULL,
    total_cost_usd   REAL    NOT NULL,
    fail_rate        REAL    NOT NULL,
    use_fake         INTEGER NOT NULL                       -- 0 / 1 (SQLite has no native bool)
);

CREATE TABLE IF NOT EXISTS answers (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL,                             -- ties an answer to its run
    question  TEXT    NOT NULL,
    answer    TEXT    NOT NULL,
    cost_usd  REAL    NOT NULL,
    retries   INTEGER NOT NULL DEFAULT 0,
    ts        REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE eval_runs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
golden_id TEXT NOT NULL,
question TEXT NOT NULL,
candidate_answer TEXT NOT NULL,
ideal_answer TEXT NOT NULL,
candidate_model TEXT NOT NULL,
judge_model TEXT NOT NULL,
accuracy INTEGER NOT NULL,
groundedness INTEGER NOT NULL,
format INTEGER NOT NULL,
reasoning TEXT NOT NULL,
eval_run_label TEXT DEFAULT 'eval-run-001',
created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: str | Path = "data/results.db") -> sqlite3.Connection:
    """Open (or create) the database, ensure both tables exist, return the connection."""
    ensure_schema(path)
    return sqlite3.connect(path)


def write_run(con: sqlite3.Connection, summary: RunSummary) -> int:
    """Insert one row into `runs`. Returns the new row id (use for write_answers)."""
    cur = con.execute(
        "INSERT INTO runs (started_at, elapsed_seconds, n_questions, n_successed, "
        "                  n_retries_total, total_cost_usd, fail_rate, use_fake) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            summary.started_at,
            summary.elapsed_seconds,
            summary.n_questions,
            summary.n_successed,
            summary.n_retries_total,
            summary.total_cost_usd,
            summary.fail_rate,
            1 if summary.use_fake else 0,
        ),
    )
    con.commit()
    return cur.lastrowid


def write_answers(
    con: sqlite3.Connection,
    run_id: int,
    answers: Iterable[Answer],
) -> int:
    """Bulk-insert all answers for a given run. Returns the number of rows inserted."""
    ts = time.time()
    rows = [(run_id, a.question, a.text, a.cost_usd, a.retries, ts) for a in answers]
    con.executemany(
        "INSERT INTO answers (run_id, question, answer, cost_usd, retries, ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    return len(rows)


def save_answer(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    question: str,
    content: str,
    retries: int,
    cost_usd: float,
    model: str,
    confidence: float,
    sources: list[str],
    schema_version: str = "v1",
) -> int:
    """Insert one row and return its rowid."""
    cur = conn.execute(
        """
        INSERT INTO answers (
            run_id, question, answer, retries, cost_usd, model,
            confidence, sources_json, schema_version, ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            question,
            content,
            retries,
            cost_usd,
            model,
            confidence,
            json.dumps(sources),
            schema_version,
            time.time(),
        ),
    )
    return cur.lastrowid


# Columns added in W4. Each entry: (column_name, column_def).
# Applied via ALTER TABLE only if missing — so the migration is safe to rerun.
_W4_NEW_COLUMNS = [
    ("model", "TEXT"),
    ("confidence", "REAL"),
    ("sources_json", "TEXT"),  # list[str] stored as JSON
    ("schema_version", "TEXT DEFAULT 'v1'"),
]


def ensure_schema(db_path: str | Path) -> None:
    """Create answers table if missing, then add any missing W4 columns.

    Safe to call on an empty file, on a W2/W3 db, or on a fully-migrated db.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        # Read current columns.
        cur = conn.execute("PRAGMA table_info(answers)")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, col_def in _W4_NEW_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE answers ADD COLUMN {col_name} {col_def}")
        conn.commit()
    finally:
        conn.close()
