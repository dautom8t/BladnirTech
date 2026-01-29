# enterprise/governance.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException


# NOTE:
# This is an in-memory store (resets on deploy/restart).
# For true enterprise persistence, we’ll later back this with your DB.
_GATE_AUTH: Dict[str, Dict[str, Any]] = {}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_gate_key(
    tenant: str,
    from_queue: str,
    to_queue: str,
    action_type: str = "advance_queue",
) -> str:
    tenant = (tenant or "default").strip().lower()
    return f"{tenant}.{from_queue}_to_{to_queue}"


def is_authorized(key: str) -> bool:
    rec = _GATE_AUTH.get(key)
    return bool(rec and rec.get("enabled") is True)


def authorize_gate(key: str, actor: str = "human", note: str = "") -> Dict[str, Any]:
    _GATE_AUTH[key] = {
        "key": key,
        "enabled": True,
        "authorized_by": actor,
        "note": note,
        "authorized_at": utcnow_iso(),
    }
    return _GATE_AUTH[key]


def revoke_gate(key: str, actor: str = "human", note: str = "") -> Dict[str, Any]:
    _GATE_AUTH[key] = {
        "key": key,
        "enabled": False,
        "revoked_by": actor,
        "note": note,
        "revoked_at": utcnow_iso(),
    }
    return _GATE_AUTH[key]


def require_authorized(key: str) -> None:
    if not is_authorized(key):
        # keep your existing error wording so the UI + logs stay consistent
        raise HTTPException(status_code=403, detail=f"Automation not authorized: {key}")


def list_gates() -> Dict[str, Dict[str, Any]]:
    return dict(_GATE_AUTH)


# Optional helper: build key from an "advance_queue" action payload
def gate_key_from_action(
    tenant: str,
    action_type: str,
    payload: Dict[str, Any],
    *,
    to_queue: Optional[str] = None,
) -> Optional[str]:
    """
    Returns gate key if the action is a queue-advance type.
    Caller may provide to_queue if it already knows it.
    """
    if action_type != "advance_queue":
        return None

    from_q = (payload or {}).get("from_queue")
    to_q = to_queue or (payload or {}).get("to_queue")
    if not from_q or not to_q:
        return None

    return make_gate_key(tenant=tenant, from_queue=from_q, to_queue=to_q)
