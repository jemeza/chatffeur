from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


ActionType = Literal["REQUESTED", "VERIFIED", "EXECUTED", "RESULT", "ERROR"]


@dataclass
class ActionLogEntry:
    action_type: ActionType
    node: str
    platform: str
    payload: dict[str, Any]
    response: dict[str, Any]
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "node": self.node,
            "platform": self.platform,
            "payload": self.payload,
            "response": self.response,
            "success": self.success,
        }
