"""
Helpers for creating and persisting action log entries.

Every tool call produces at least one entry describing:
  requested — what the agent asked for
  verified  — what was checked before execution
  executed  — what was actually passed to the adapter / external system
  outcome   — human-readable result ("success", "error: …", "confirmed", …)

Entries are appended to action_log.jsonl for durable, queryable history
and also returned so LangGraph can merge them into AgentState.action_log.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("action_log.jsonl")


def make_log_entry(
    action_type: str,
    requested: dict,
    verified: dict,
    executed: dict,
    outcome: str,
) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_type": action_type,
        "requested": requested,
        "verified": verified,
        "executed": executed,
        "outcome": outcome,
    }
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
