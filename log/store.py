from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from log.models import ActionLogEntry, ActionType


_DB_PATH = os.getenv("LOG_DB_PATH", "chatffeur.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS action_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    action_type TEXT    NOT NULL,
    node        TEXT    NOT NULL,
    platform    TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    success     INTEGER NOT NULL
)
"""


@contextmanager
def _conn():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str | None = None) -> None:
    global _DB_PATH
    if db_path:
        _DB_PATH = db_path
    with _conn() as con:
        con.execute(_CREATE_TABLE)


def append(entry: ActionLogEntry) -> ActionLogEntry:
    """Persist one log entry and return it with its auto-assigned id."""
    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO action_log (timestamp, action_type, node, platform, payload, response, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp.isoformat(),
                entry.action_type,
                entry.node,
                entry.platform,
                json.dumps(entry.payload),
                json.dumps(entry.response),
                int(entry.success),
            ),
        )
        entry.id = cur.lastrowid
    return entry


def fetch_all(limit: int = 200) -> list[ActionLogEntry]:
    """Return the most recent log entries, newest last."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM action_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_entry(r) for r in reversed(rows)]


def fetch_since(entry_id: int) -> list[ActionLogEntry]:
    """Return entries with id > entry_id (for polling new entries only)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM action_log WHERE id > ? ORDER BY id ASC", (entry_id,)
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: sqlite3.Row) -> ActionLogEntry:
    return ActionLogEntry(
        id=row["id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        action_type=row["action_type"],
        node=row["node"],
        platform=row["platform"],
        payload=json.loads(row["payload"]),
        response=json.loads(row["response"]),
        success=bool(row["success"]),
    )


# ---------------------------------------------------------------------------
# Decorator: wrap any node function and auto-log REQUESTED + RESULT/ERROR
# ---------------------------------------------------------------------------

def log_action(node_name: str, platform_key: str = "platform"):
    """
    Decorator for LangGraph node functions.

    The wrapped function must accept state as its first positional arg.
    platform_key is the attribute on state that holds the platform name.

    Usage:
        @log_action("discover_options")
        def discover_options(state: AgentState) -> dict: ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(state: Any, *args, **kwargs) -> Any:
            platform = getattr(state, platform_key, "unknown")
            payload = {
                "pickup": getattr(state, "pickup", None),
                "dropoff": getattr(state, "dropoff", None),
            }
            append(ActionLogEntry(
                action_type="REQUESTED",
                node=node_name,
                platform=platform,
                payload=payload,
                response={},
                success=True,
            ))
            try:
                result = fn(state, *args, **kwargs)
                append(ActionLogEntry(
                    action_type="RESULT",
                    node=node_name,
                    platform=platform,
                    payload=payload,
                    response=_serialise(result),
                    success=True,
                ))
                return result
            except Exception as exc:
                append(ActionLogEntry(
                    action_type="ERROR",
                    node=node_name,
                    platform=platform,
                    payload=payload,
                    response={"error": str(exc)},
                    success=False,
                ))
                raise
        return wrapper
    return decorator


def log_verification(node: str, platform: str, details: dict) -> None:
    """Log a user verification event (confirmation gate answer)."""
    append(ActionLogEntry(
        action_type="VERIFIED",
        node=node,
        platform=platform,
        payload=details,
        response={},
        success=True,
    ))


def log_executed(node: str, platform: str, payload: dict, response: dict) -> None:
    """Log an explicit execution event (e.g., booking placed)."""
    append(ActionLogEntry(
        action_type="EXECUTED",
        node=node,
        platform=platform,
        payload=payload,
        response=response,
        success=True,
    ))


def _serialise(obj: Any) -> dict:
    """Best-effort serialisation of a node return value for logging."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return {"raw": str(obj)}
    return {"raw": str(obj)}
