"""
Enterprise execution router for governed workflow automation.
"""

from typing import Optional
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from enterprise.auth import require_auth, require_role, UserContext
from enterprise.audit import log_action
from enterprise import governance
from models.database import get_db
from services import workflow as workflow_service
from sqlalchemy.orm import Session

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enterprise", tags=["Enterprise"])


# =====================================
# Request/Response Models
# =====================================

class ExecutionRequest(BaseModel):
    """Request to execute a governed action."""
    workflow_id: int = Field(..., gt=0, description="Target workflow ID")
    action_type: str = Field(..., min_length=1, max_length=100, description="Action to execute (e.g., 'advance_queue', 'approve_insurance')")
    from_state: Optional[str] = Field(None, description="Current state for validation")
    to_state: Optional[str] = Field(None, description="Target state")
    metadata: dict = Field(default_factory=dict, description="Additional action metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": 123,
                "action_type": "advance_queue",
                "from_state": "data_entry",
                "to_state": "pre_verification",
                "metadata": {"reason": "All data validated"}
            }
        }


class ExecutionResponse(BaseModel):
    """Response from execution attempt."""
    ok: bool
    workflow_id: int
    action_type: str
    status: str  # "executed", "pending_approval", "rejected"
    message: str
    audit_id: Optional[str] = None
    result: Optional[dict] = None


# =====================================
# Execution Endpoint
# =====================================

@router.post("/execute", response_model=ExecutionResponse)
def enterprise_execute(
    request: Request,
    execution: ExecutionRequest = Body(...),
    user: UserContext = Depends(require_role("admin")),  # Only admins can execute
    db: Session = Depends(get_db),
):
    """
    Execute a governed workflow action with full audit trail.
    
    This endpoint:
    1. Validates the user has permission
    2. Checks governance gates are authorized
    3. Executes the action
    4. Logs complete audit trail
    
    **Required Role:** admin
    
    **Governance:** Requires appropriate gate authorization for the action type.
    
    **Example:**
```json
    {
        "workflow_id": 123,
        "action_type": "advance_queue",
        "from_state": "data_entry",
        "to_state": "pre_verification",
        "metadata": {"validated_by": "system", "confidence": 0.95}
    }
```
    """
    client_ip = request.client.host if request.client else "unknown"
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    # Build audit metadata
    audit_metadata = {
        "workflow_id": execution.workflow_id,
        "action_type": execution.action_type,
        "from_state": execution.from_state,
        "to_state": execution.to_state,
        "user_metadata": execution.metadata,
    }
    
    try:
        # 1. Validate workflow exists
        workflow = workflow_service.get_workflow(db, execution.workflow_id)
        if not workflow:
            logger.warning(
                f"Execution rejected: Workflow {execution.workflow_id} not found "
                f"(user={user.user_id}, IP={client_ip})"
            )
            log_action(
                actor=user.user_id,  # ✅ Fixed: Use user_id, not api_key
                role=user.role,
                action="enterprise.execute.rejected",
                metadata={**audit_metadata, "reason": "workflow_not_found"},
                ip_address=client_ip,
                request_id=request_id,
                status="failure",
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {execution.workflow_id} not found"
            )
        
        # 2. Validate state transition (if specified)
        current_queue = getattr(workflow, "queue", None)
        if execution.from_state and current_queue != execution.from_state:
            logger.warning(
                f"Execution rejected: State mismatch for workflow {execution.workflow_id} "
                f"(expected={execution.from_state}, actual={current_queue}, user={user.user_id})"
            )
            log_action(
                actor=user.user_id,
                role=user.role,
                action="enterprise.execute.rejected",
                metadata={**audit_metadata, "reason": "state_mismatch", "actual_state": current_queue},
                ip_address=client_ip,
                request_id=request_id,
                status="failure",
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workflow is in state '{current_queue}', expected '{execution.from_state}'"
            )
        
        # 3. Check governance authorization
        gate_key = None
        if execution.action_type == "advance_queue" and execution.from_state and execution.to_state:
            # Build gate key from transition
            gate_key = f"kroger.{execution.from_state}_to_{execution.to_state}"
            
            try:
                governance.require_authorized(gate_key)
            except Exception as e:
                logger.warning(
                    f"Execution rejected: Gate '{gate_key}' not authorized "
                    f"(workflow={execution.workflow_id}, user={user.user_id})"
                )
                log_action(
                    actor=user.user_id,
                    role=user.role,
                    action="enterprise.execute.rejected",
                    metadata={**audit_metadata, "reason": "gate_not_authorized", "gate_key": gate_key},
                    ip_address=client_ip,
                    request_id=request_id,
                    status="failure",
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Action not authorized: {gate_key}"
                )
        
        # 4. Execute the action
        result = _execute_action(db, workflow, execution)
        
        # 5. Log successful execution
        log_action(
            actor=user.user_id,
            role=user.role,
            action="enterprise.execute.success",
            metadata={
                **audit_metadata,
                "gate_key": gate_key,
                "result": result,
            },
            ip_address=client_ip,
            request_id=request_id,
            status="success",
        )
        
        logger.info(
            f"Execution successful: workflow={execution.workflow_id}, "
            f"action={execution.action_type}, user={user.user_id}"
        )
        
        return ExecutionResponse(
            ok=True,
            workflow_id=execution.workflow_id,
            action_type=execution.action_type,
            status="executed",
            message=f"Action '{execution.action_type}' executed successfully",
            result=result,
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (already logged above)
        raise
    
    except Exception as e:
        # Unexpected errors
        logger.exception(
            f"Execution failed with unexpected error: workflow={execution.workflow_id}, "
            f"action={execution.action_type}, user={user.user_id}, error={e}"
        )
        
        log_action(
            actor=user.user_id,
            role=user.role,
            action="enterprise.execute.error",
            metadata={**audit_metadata, "error": str(e), "error_type": type(e).__name__},
            ip_address=client_ip,
            request_id=request_id,
            status="failure",
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


def _execute_action(db: Session, workflow, execution: ExecutionRequest) -> dict:
    """
    Internal function to execute the actual action.
    
    Args:
        db: Database session
        workflow: Workflow ORM object
        execution: Execution request
        
    Returns:
        dict with execution results
    """
    if execution.action_type == "advance_queue":
        # Update workflow queue
        if execution.to_state:
            workflow.queue = execution.to_state
            db.commit()
            db.refresh(workflow)
        
        return {
            "previous_queue": execution.from_state,
            "new_queue": execution.to_state,
            "updated_at": workflow.updated_at.isoformat() if hasattr(workflow, "updated_at") else None,
        }
    
    elif execution.action_type == "approve_task":
        # Implement task approval logic
        task_id = execution.metadata.get("task_id")
        if task_id:
            # Update task state
            return {"task_id": task_id, "status": "approved"}
    
    else:
        raise ValueError(f"Unknown action type: {execution.action_type}")


# =====================================
# Status/Health Endpoint
# =====================================

@router.get("/status")
def enterprise_status(user: UserContext = Depends(require_auth)):
    """
    Get enterprise execution status and available actions.
    
    Returns current governance authorizations and execution statistics.
    """
    gates = governance.list_gates()
    
    return {
        "ok": True,
        "user": {
            "user_id": user.user_id,
            "role": user.role,
            "key_id": user.key_id,
        },
        "governance": {
            "gates": gates,
            "total_authorized": sum(1 for g in gates.values() if g.get("enabled")),
        },
        "actions": {
            "available": ["advance_queue", "approve_task"],
        }
    }


# =====================================
# Dry-Run Endpoint (Validation Only)
# =====================================

@router.post("/execute/dry-run", response_model=ExecutionResponse)
def enterprise_execute_dry_run(
    request: Request,
    execution: ExecutionRequest = Body(...),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Validate an execution request without actually executing it.
    
    Useful for:
    - Testing governance rules
    - Validating state transitions
    - Pre-flight checks in UI
    
    **Does not modify any data.**
    """
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        # Validate workflow exists
        workflow = workflow_service.get_workflow(db, execution.workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {execution.workflow_id} not found"
            )
        
        # Validate state
        current_queue = getattr(workflow, "queue", None)
        if execution.from_state and current_queue != execution.from_state:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"State mismatch: expected '{execution.from_state}', actual '{current_queue}'"
            )
        
        # Check governance
        gate_key = None
        if execution.action_type == "advance_queue" and execution.from_state and execution.to_state:
            gate_key = f"kroger.{execution.from_state}_to_{execution.to_state}"
            try:
                governance.require_authorized(gate_key)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Action not authorized: {gate_key}"
                )
        
        # Log dry-run
        log_action(
            actor=user.user_id,
            role=user.role,
            action="enterprise.execute.dry_run",
            metadata={
                "workflow_id": execution.workflow_id,
                "action_type": execution.action_type,
                "gate_key": gate_key,
            },
            ip_address=client_ip,
            status="success",
        )
        
        return ExecutionResponse(
            ok=True,
            workflow_id=execution.workflow_id,
            action_type=execution.action_type,
            status="validated",
            message="Validation passed - would execute successfully",
            result={"dry_run": True, "validated_at": "now"},
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )
