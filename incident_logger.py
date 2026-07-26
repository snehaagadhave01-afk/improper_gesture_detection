"""
incident_logger.py
--------------------
Lightweight SQLite log of every alert the system raises, so you get a
persistent history and can run analytics later -- addresses "existing
monitoring systems have no historical record or analytical insights."

No extra dependency -- sqlite3 is part of the Python standard library.
"""

import sqlite3
import time
from contextlib import closing

import config


def _connect():
    return sqlite3.connect(config.INCIDENTS_DB_PATH)


def init_db():
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                snapshot_path TEXT
            )
            """
        )
        conn.commit()


def log_incident(incident_type: str, message: str, snapshot_path: str = None):
    if not config.ENABLE_INCIDENT_LOGGING:
        return
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO incidents (timestamp, type, message, snapshot_path) VALUES (?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), incident_type, message, snapshot_path),
        )
        conn.commit()


def fetch_recent(limit: int = 20):
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]


def counts_by_type():
    with closing(_connect()) as conn:
        cur = conn.execute(
            "SELECT type, COUNT(*) as count FROM incidents GROUP BY type ORDER BY count DESC"
        )
        return cur.fetchall()


def total_count():
    with closing(_connect()) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM incidents")
        return cur.fetchone()[0]