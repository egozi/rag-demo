import sqlite3
from pathlib import Path


class ConversationHistory:
    def __init__(self, db_path: str | Path = "data/history.db"):
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id  TEXT PRIMARY KEY,
                    summary     TEXT NOT NULL,
                    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "user", question),
            )
            conn.execute(
                "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "assistant", answer),
            )

    def get_context(self, session_id: str) -> str:
        from api.config import get_settings

        settings = get_settings()
        with self._connect() as conn:
            summary_row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
            summary = summary_row["summary"] if summary_row else ""

            recent_rows = conn.execute(
                """SELECT role, content FROM turns
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, settings.short_term_turns * 2),
            ).fetchall()

        if not summary and not recent_rows:
            return ""

        parts: list[str] = []
        if summary:
            parts.append(f"[Summary of earlier conversation]\n{summary}")

        if recent_rows:
            recent_text = "\n".join(
                f"{r['role'].capitalize()}: {r['content']}"
                for r in reversed(recent_rows)
            )
            parts.append(f"[Recent conversation]\n{recent_text}")

        return "\n\n".join(parts)

    def maybe_summarize(self, session_id: str, llm_client) -> None:
        from api.config import get_settings

        settings = get_settings()
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

            if total <= settings.long_term_threshold:
                return

            rows_to_summarize = total - settings.short_term_turns * 2
            if rows_to_summarize <= 0:
                return

            old_rows = conn.execute(
                """SELECT id, role, content FROM turns
                   WHERE session_id = ?
                   ORDER BY id ASC LIMIT ?""",
                (session_id, rows_to_summarize),
            ).fetchall()

        if not old_rows:
            return

        # Get existing summary to incorporate
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        prior_summary = existing["summary"] if existing else ""

        turns_text = "\n".join(
            f"{r['role'].capitalize()}: {r['content']}" for r in old_rows
        )
        prompt_parts = []
        if prior_summary:
            prompt_parts.append(f"Prior summary:\n{prior_summary}\n")
        prompt_parts.append(f"New conversation turns to summarize:\n{turns_text}")

        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation turns into a concise paragraph "
                    "that preserves the key facts and topics discussed. "
                    "Integrate any prior summary provided."
                ),
            },
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]
        new_summary = llm_client.chat(messages, span_name="history-summarize")

        old_ids = [r["id"] for r in old_rows]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO summaries (session_id, summary, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary = excluded.summary,
                       updated_at = excluded.updated_at""",
                (session_id, new_summary),
            )
            conn.execute(
                f"DELETE FROM turns WHERE id IN ({','.join('?' * len(old_ids))})",
                old_ids,
            )

    def clear(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
