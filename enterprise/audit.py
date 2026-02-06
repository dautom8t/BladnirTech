"""
Enterprise-grade audit logging with proper error handling, rotation, and compliance features.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Configuration
AUDIT_DIR = Path(os.getenv("AUDIT_LOG_DIR", "./logs/audit"))
AUDIT_FILE = AUDIT_DIR / "audit_log.jsonl"
MAX_LOG_SIZE = int(os.getenv("AUDIT_MAX_SIZE_MB", "100")) * 1024 * 1024  # Default 100MB

# Thread-safe file lock
_lock = threading.Lock()


def _ensure_audit_directory():
    """Ensure audit log directory exists with proper permissions."""
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        # Set restrictive permissions (owner read/write only)
        os.chmod(AUDIT_DIR, 0o700)
    except Exception as e:
        logger.error(f"Failed to create audit directory: {e}")
        raise


def _rotate_if_needed():
    """Rotate log file if it exceeds max size."""
    try:
        if AUDIT_FILE.exists() and AUDIT_FILE.stat().st_size >= MAX_LOG_SIZE:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated_file = AUDIT_DIR / f"audit_log_{timestamp}.jsonl"
            AUDIT_FILE.rename(rotated_file)
            logger.info(f"Rotated audit log to {rotated_file}")
    except Exception as e:
        logger.error(f"Failed to rotate audit log: {e}")
        # Don't raise - allow logging to continue


def log_action(
    actor: str,
    role: str,
    action: str,
    metadata: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = "success",
) -> bool:
    """
    Log an auditable action with comprehensive metadata.
    
    Args:
        actor: User identifier (username, email, service account)
        role: User's role or permission level
        action: Action performed (e.g., "workflow.create", "proposal.approve")
        metadata: Additional context (workflow_id, changes, etc.)
        ip_address: Source IP address
        request_id: Correlation ID for request tracing
        status: "success" or "failure"
    
    Returns:
        bool: True if logged successfully, False otherwise
        
    Note:
        This function never raises exceptions to avoid disrupting application flow.
    """
    # Validate required fields
    if not actor or not role or not action:
        logger.error("Audit log rejected: missing required fields")
        return False
    
    # Sanitize inputs (basic validation)
    actor = str(actor)[:100]
    role = str(role)[:50]
    action = str(action)[:100]
    
    # Build entry with ISO 8601 timestamp
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix": int(datetime.now(timezone.utc).timestamp()),
        "actor": actor,
        "role": role,
        "action": action,
        "status": status,
        "metadata": metadata or {},
    }
    
    # Add optional fields
    if ip_address:
        entry["ip_address"] = str(ip_address)[:45]  # IPv6 max length
    if request_id:
        entry["request_id"] = str(request_id)[:100]
    
    # Thread-safe write with error handling
    try:
        with _lock:
            # Ensure directory exists
            _ensure_audit_directory()
            
            # Rotate if needed
            _rotate_if_needed()
            
            # Write entry
            with open(AUDIT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()  # Ensure immediate write to disk
                os.fsync(f.fileno())  # Force OS to write to physical disk
        
        return True
        
    except PermissionError:
        logger.error(f"Permission denied writing to audit log: {AUDIT_FILE}")
        return False
    except OSError as e:
        logger.error(f"OS error writing to audit log: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error writing to audit log: {e}")
        return False


def log_governance_decision(
    actor: str,
    role: str,
    gate_key: str,
    decision: str,
    note: str = "",
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
) -> bool:
    """
    Specialized logging for governance/authorization decisions.
    
    Args:
        actor: User making the decision
        role: User's role
        gate_key: Governance gate being modified (e.g., "kroger.data_entry_to_preverify")
        decision: "authorize" or "revoke"
        note: Justification for the decision
        ip_address: Source IP
        request_id: Request correlation ID
    
    Returns:
        bool: True if logged successfully
    """
    return log_action(
        actor=actor,
        role=role,
        action=f"governance.{decision}",
        metadata={
            "gate_key": gate_key,
            "decision": decision,
            "note": note,
        },
        ip_address=ip_address,
        request_id=request_id,
    )


def log_proposal_decision(
    actor: str,
    role: str,
    proposal_id: str,
    decision: str,
    case_id: int,
    action_type: str,
    note: str = "",
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
) -> bool:
    """
    Specialized logging for proposal approve/reject decisions.
    
    Args:
        actor: User making the decision
        role: User's role
        proposal_id: Proposal identifier
        decision: "approve" or "reject"
        case_id: Associated case/workflow ID
        action_type: Type of action being approved (e.g., "advance_queue")
        note: Justification
        ip_address: Source IP
        request_id: Request correlation ID
    
    Returns:
        bool: True if logged successfully
    """
    return log_action(
        actor=actor,
        role=role,
        action=f"proposal.{decision}",
        metadata={
            "proposal_id": proposal_id,
            "case_id": case_id,
            "action_type": action_type,
            "decision": decision,
            "note": note,
        },
        ip_address=ip_address,
        request_id=request_id,
    )


# Optional: Async version for high-throughput applications
try:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audit")
    
    async def log_action_async(
        actor: str,
        role: str,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        status: str = "success",
    ) -> bool:
        """
        Async version of log_action for use in async FastAPI endpoints.
        Offloads I/O to thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            log_action,
            actor,
            role,
            action,
            metadata,
            ip_address,
            request_id,
            status,
        )

except ImportError:
    # asyncio not available, skip async version
    pass
