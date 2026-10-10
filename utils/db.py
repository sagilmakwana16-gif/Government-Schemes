import json
import os
import sqlite3
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger("db")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "conversations.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Ensures parent directory exists.
    """
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.exception("Failed to connect to SQLite database at '%s': %s", db_path, e)
        raise


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database tables and indexes if they do not exist.
    """
    try:
        logger.info("Initializing SQLite database at '%s'...", db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id_created 
                ON conversation_history(session_id, created_at)
            """)
            conn.commit()
        logger.info("SQLite database tables and indexes verified successfully.")
    except Exception as e:
        logger.exception("Failed to initialize SQLite database at '%s': %s", db_path, e)
        raise


def save_conversation(
    session_id: str,
    question: str,
    answer: str,
    sources: Optional[List[str]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Saves a Q&A conversation entry for a given session.
    Returns the newly inserted row ID.
    """
    try:
        sources_json = json.dumps(sources or [])
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_history (session_id, question, answer, sources)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, question.strip(), answer.strip(), sources_json)
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(
                "Saved conversation entry [ID: %s] for session '%s' (question: %.60s...)",
                row_id, session_id, question.strip()
            )
            return row_id
    except Exception as e:
        logger.exception("Failed to save conversation for session '%s': %s", session_id, e)
        raise


def get_last_n_qa_pairs(
    session_id: str,
    limit: int = 5,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, str]]:
    """
    Retrieves the last N Q&A pairs for a given session in chronological order (oldest first).
    Returns a list of dicts: [{'question': str, 'answer': str}, ...]
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT question, answer FROM (
                    SELECT id, question, answer
                    FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            pairs = [{"question": row["question"], "answer": row["answer"]} for row in rows]
            logger.debug(
                "Retrieved %d recent QA pair(s) (limit=%d) for session '%s'",
                len(pairs), limit, session_id
            )
            return pairs
    except Exception as e:
        logger.exception("Failed to retrieve QA pairs for session '%s': %s", session_id, e)
        raise


def get_session_history(
    session_id: str,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves full message history for a session.
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, question, answer, sources, created_at
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            history = []
            for row in rows:
                try:
                    sources_list = json.loads(row["sources"]) if row["sources"] else []
                except Exception:
                    sources_list = []
                history.append({
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": sources_list,
                    "created_at": row["created_at"]
                })
            logger.info("Retrieved %d history records for session '%s'", len(history), session_id)
            return history
    except Exception as e:
        logger.exception("Failed to retrieve session history for session '%s': %s", session_id, e)
        raise


def clear_session_history(session_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Clears all conversation records for a given session.
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info("Cleared %d conversation history record(s) for session '%s'", deleted_count, session_id)
            return True
    except Exception as e:
        logger.exception("Failed to clear conversation history for session '%s': %s", session_id, e)
        raise