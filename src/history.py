from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DB_PATH


class HistoryStore:
    """Store classification history in SQLite."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    bin_label TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_record(self, category: str, confidence: float, bin_label: str, image_path: str | None = None):
        from datetime import datetime

        timestamp = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO history (timestamp, category, confidence, image_path, bin_label) VALUES (?, ?, ?, ?, ?)",
                (timestamp, category, confidence, image_path, bin_label),
            )
            conn.commit()

    def fetch_all(self):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT timestamp, category, confidence, image_path, bin_label FROM history ORDER BY id DESC"
            ).fetchall()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()
