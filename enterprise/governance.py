"""
Enterprise governance system with persistent storage, audit logging, and RBAC.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException

from enterprise.audit import log_action

logger = logging.getLogger(__name__)

# =====================================
# Configuration
# =====================================

GOVERNANCE_FILE = Path("data/governance_gates.json")
GOVERNANCE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Thread-safe lock for file I/O
_lock = threading.Lock()


# =====================================
# Data Models
# =====================================

@dataclass
class GateChange:
    """Record of a single authorization/revocation."""
    timestamp: str
    actor: str
    action: str  # "authorize" or "revoke"
    note: str
    enabled: bool


@dataclass
class Gate:
    """Governance gate with full history."""
    key: str
    enabled: bool
    created_at: str
    updated_at: str
    history: List[GateChange]
    
    @property
    def current_actor(self) -> Optional[str]:
        """Get the actor who made the most recent change."""
        return self.history[-1].actor if self.history else None
    
    @property
    def current_note(self) -> Optional[str]:
        """Get the note from the most recent change."""
        return self.history[-1].note if self.history else None


# =====================================
# In-Memory Storage + Persistence
# =====================================

_GATES: Dict[str, Gate] = {}


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _load_gates() -> None:
    """Load gates from persistent storage on startup."""
    global _GATES
    
    if not GOVERNANCE_FILE.exists():
        logger.info("No governance file found, starting with empty gates")
        return
    
    try:
        with open(GOVERNANCE_FILE, "r") as f:
            data = json.load(f)
        
        # Deserialize gates
        for key, gate_data in data.items():
            history = [GateChange(**change) for change in gate_data.get("history", [])]
            _GATES[key] = Gate(
                key=gate_data["key"],
                enabled=gate_data["enabled"],
                created_at=gate_data["created_at"],
                updated_at=gate_data["updated_at"],
                history=history,
            )
        
        logger.info(f"Loaded {len(_GATES)} governance gates from {GOVERNANCE_FILE}")
    
    except Exception as e:
        logger.error(f"Failed to load governance gates: {e}")
        # Don't crash - start with empty gates


def _save_gates() -> None:
    """Persist gates to disk."""
    try:
        # Serialize gates
        data = {}
        for key, gate in _GATES.items():
            data[key] = {
                "key": gate.key,
                "enabled": gate.enabled,
                "created_at": gate.created_at,
                "updated_at": gate.updated_at,
                "history": [asdict(change) for change in gate.history],
            }
        
        # Write atomically (write to temp file, then rename)
        temp_file = GOVERNANCE_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        
        temp_file.replace(GOVERNANCE_FILE)
        logger.debug(f"Saved {len(_GATES)} gates to {GOVERNANCE_FILE}")
    
    except Exception as e:
        logger.error(f"Failed to save governance gates: {e}")


# Load gates on module import
_load_gates()


# =====================================
# Core Functions
# =====================================

def is_authorized(key: str) -> bool:
    """
    Check if a gate is currently authorized.
    
    Args:
        key: Gate key to check
        
    Returns:
        True if gate exists and is enabled, False otherwise
    """
    if not key:
        return False
    
    with _lock:
        gate = _GATES.get(key)
        return gate.enabled if gate else False


def authorize_gate(
    key: str,
    actor: str = "human",
    note: str = "",
    ip_address: Optional[str] = None,
) -> Gate:
    """
    Authorize a governance gate.
    
    Args:
        key: Gate identifier (e.g., "kroger.data_entry_to_preverify")
        actor: User/system authorizing the gate
        note: Justification for authorization
        ip_address: IP address of requester (for audit)
        
    Returns:
        Updated Gate object
        
    Raises:
        ValueError: If key/actor invalid
    """
    # Validate inputs
    if not key or len(key) > 200:
        raise ValueError("Gate key must be 1-200 characters")
    if not actor or len(actor) > 100:
        raise ValueError("Actor must be 1-100 characters")
    if len(note) > 500:
        raise ValueError("Note must be <= 500 characters")
    
    now = _now_iso()
    
    with _lock:
        # Get or create gate
        gate = _GATES.get(key)
        
        if gate is None:
            # Create new gate
            gate = Gate(
                key=key,
                enabled=True,
                created_at=now,
                updated_at=now,
                history=[],
            )
            _GATES[key] = gate
            logger.info(f"Created new gate: {key}")
        
        # Add history entry
        change = GateChange(
            timestamp=now,
            actor=actor,
            action="authorize",
            note=note,
            enabled=True,
        )
        gate.history.append(change)
        
        # Update gate state
        gate.enabled = True
        gate.updated_at = now
        
        # Persist to disk
        _save_gates()
    
    # Audit log (outside lock to avoid deadlock)
    log_action(
        actor=actor,
        role="admin",  # Only admins should authorize
        action="governance.authorize",
        metadata={
            "gate_key": key,
            "note": note,
        },
        ip_address=ip_address,
    )
    
    logger.info(f"Gate authorized: {key} by {actor}")
    return gate


def revoke_gate(
    key: str,
    actor: str = "human",
    note: str = "",
    ip_address: Optional[str] = None,
) -> Gate:
    """
    Revoke a governance gate.
    
    Args:
        key: Gate identifier
        actor: User/system revoking the gate
        note: Justification for revocation
        ip_address: IP address of requester (for audit)
        
    Returns:
        Updated Gate object
        
    Raises:
        ValueError: If key/actor invalid
    """
    # Validate inputs
    if not key or len(key) > 200:
        raise ValueError("Gate key must be 1-200 characters")
    if not actor or len(actor) > 100:
        raise ValueError("Actor must be 1-100 characters")
    if len(note) > 500:
        raise ValueError("Note must be <= 500 characters")
    
    now = _now_iso()
    
    with _lock:
        # Get or create gate
        gate = _GATES.get(key)
        
        if gate is None:
            # Create gate in revoked state
            gate = Gate(
                key=key,
                enabled=False,
                created_at=now,
                updated_at=now,
                history=[],
            )
            _GATES[key] = gate
        
        # Add history entry
        change = GateChange(
            timestamp=now,
            actor=actor,
            action="revoke",
            note=note,
            enabled=False,
        )
        gate.history.append(change)
        
        # Update gate state
        gate.enabled = False
        gate.updated_at = now
        
        # Persist to disk
        _save_gates()
    
    # Audit log
    log_action(
        actor=actor,
        role="admin",
        action="governance.revoke",
        metadata={
            "gate_key": key,
            "note": note,
        },
        ip_address=ip_address,
    )
    
    logger.info(f"Gate revoked: {key} by {actor}")
    return gate


def require_authorized(key: str, raise_exception: bool = True) -> bool:
    """
    Require a gate to be authorized, raising exception if not.
    
    Args:
        key: Gate key to check
        raise_exception: If True, raises HTTPException on failure
        
    Returns:
        True if authorized
        
    Raises:
        HTTPException: 403 if not authorized and raise_exception=True
    """
    if not key:
        if raise_exception:
            raise HTTPException(
                status_code=403,
                detail="Automation gate not specified"
            )
        return False
    
    authorized = is_authorized(key)
    
    if not authorized and raise_exception:
        logger.warning(f"Unauthorized gate access attempt: {key}")
        raise HTTPException(
            status_code=403,
            detail="This automation requires explicit authorization. Please enable it in the dashboard."
        )
    
    return authorized


def list_gates() -> Dict[str, Dict[str, any]]:
    """
    List all gates with their current state and metadata.
    
    Returns:
        Dictionary of gate keys to gate information
    """
    with _lock:
        return {
            key: {
                "key": gate.key,
                "enabled": gate.enabled,
                "created_at": gate.created_at,
                "updated_at": gate.updated_at,
                "current_actor": gate.current_actor,
                "current_note": gate.current_note,
                "change_count": len(gate.history),
            }
            for key, gate in _GATES.items()
        }


def get_gate_history(key: str) -> List[Dict[str, any]]:
    """
    Get full history for a specific gate.
    
    Args:
        key: Gate key
        
    Returns:
        List of changes in chronological order
        
    Raises:
        HTTPException: 404 if gate not found
    """
    with _lock:
        gate = _GATES.get(key)
        if not gate:
            raise HTTPException(
                status_code=404,
                detail=f"Gate not found: {key}"
            )
        
        return [asdict(change) for change in gate.history]


def reset_all_gates(actor: str = "system", note: str = "Emergency reset") -> int:
    """
    Emergency function to revoke all gates.

    Args:
        actor: User/system performing reset
        note: Justification

    Returns:
        Number of gates revoked
    """
    # Collect keys to revoke under the lock, then revoke outside it.
    # revoke_gate() acquires _lock internally, so holding _lock here
    # would deadlock (threading.Lock is not reentrant).
    with _lock:
        keys_to_revoke = [key for key, gate in _GATES.items() if gate.enabled]

    count = 0
    for key in keys_to_revoke:
        revoke_gate(key, actor=actor, note=note)
        count += 1

    logger.warning(f"Emergency reset: {count} gates revoked by {actor}")
    return count


# =====================================
# Predefined Gates (Optional)
# =====================================

# Define known gates for documentation/validation
KNOWN_GATES = {
    "kroger.prescriber_approval_to_data_entry": "Allow automated transition from prescriber approval to data entry",
    "kroger.data_entry_to_preverify_insurance": "Allow automated transition from data entry to pre-verification",
    "kroger.preverify_to_access_granted": "Allow automated transition from pre-verification to access granted",
}


def validate_gate_key(key: str) -> bool:
    """
    Check if a gate key is recognized/valid.
    
    Args:
        key: Gate key to validate
        
    Returns:
        True if key is in KNOWN_GATES (or if KNOWN_GATES is empty - open validation)
    """
    if not KNOWN_GATES:
        return True  # Open validation if no known gates defined
    
    return key in KNOWN_GATES
