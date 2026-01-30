# enterprise/governance.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import HTTPException

_GATES: Dict[str, Dict[str, Any]] = {}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_authorized(key: str) -> bool:
    rec = _GATES.get(key)
    return bool(rec and rec.get("enabled") is True)

def authorize_gate(key: str, actor: str = "human", note: str = "") -> Dict[str, Any]:
    rec = _GATES.get(key) or {"key": key}
    rec.update({
        "enabled": True,
        "authorized_by": actor,
        "note": note,
        "authorized_at": _now_iso(),
    })
    _GATES[key] = rec
    return rec

def revoke_gate(key: str, actor: str = "human", note: str = "") -> Dict[str, Any]:
    rec = _GATES.get(key) or {"key": key}
    rec.update({
        "enabled": False,
        "revoked_by": actor,
        "note": note,
        "revoked_at": _now_iso(),
    })
    _GATES[key] = rec
    return rec

def require_authorized(key: str) -> None:
    if not is_authorized(key):
        raise HTTPException(status_code=403, detail=f"Automation not authorized: {key}")

def list_gates() -> Dict[str, Dict[str, Any]]:
    return dict(_GATES)

