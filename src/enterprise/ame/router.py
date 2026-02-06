"""
AME API - Adaptive Model Evolution REST endpoints
Manages trust scoring, stage progression, and execution tracking.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Request, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from models.database import get_db
from enterprise.auth import require_auth, require_role, UserContext
from enterprise.audit import log_action

from .service import (
    AMEConfig,
    log_event,
    get_scope,
    compute_and_update_scope,
    resolve_execution_mode,
    create_execution,
    rollback_execution,
)
from .models import AMEEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ame", tags=["AME - Adaptive Model Evolution"])


# =====================================
# Request Schemas
# =====================================

class ScopeKey(BaseModel):
    """Identifies a specific trust scope."""
    tenant_id: str = Field(default="default", description="Tenant identifier")
    site_id: str = Field(..., min_length=1, max_length=120, description="Site identifier")
    queue: str = Field(..., min_length=1, max_length=120, description="Queue name")
    action_type: str = Field(..., min_length=1, max_length=120, description="Action type")
    role_context: Optional[str] = Field(None, max_length=120, description="Optional role context")

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "acme_corp",
                "site_id": "pharmacy_01",
                "queue": "data_entry",
                "action_type": "advance_to_verification",
                "role_context": "pharmacist"
            }
        }


class ProposalEventIn(BaseModel):
    """AI proposes an automated action."""
    scope: ScopeKey
    proposal_id: Optional[str] = Field(None, max_length=64, description="Unique proposal ID")

    predicted_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Model's confidence in this action (0.0-1.0)"
    )
    predicted_safety: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Model's safety assessment (0.0-1.0)"
    )
    predicted_time_saved_sec: Optional[float] = Field(
        None,
        ge=0.0,
        description="Predicted time savings in seconds"
    )

    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input features used for decision"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual information (load, SLA, etc.)"
    )


class DecisionEventIn(BaseModel):
    """Human decides on AI proposal."""
    scope: ScopeKey
    proposal_id: Optional[str] = Field(None, max_length=64)
    decision: str = Field(..., pattern="^(approve|reject|defer)$", description="approve, reject, or defer")
    decision_reason: Optional[str] = Field(None, max_length=500, description="Justification for decision")


class ExecutionEventIn(BaseModel):
    """Action is executed (possibly with rollback window)."""
    scope: ScopeKey
    proposal_id: Optional[str] = Field(None, max_length=64)

    before_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="State before execution (for rollback)"
    )
    after_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="State after execution"
    )
    guarded: bool = Field(
        default=True,
        description="If true, creates rollback window"
    )


class OutcomeEventIn(BaseModel):
    """Observed outcome of executed action."""
    scope: ScopeKey
    proposal_id: Optional[str] = Field(None, max_length=64)

    outcome_success: bool = Field(default=True, description="Was the action successful?")
    observed_error: bool = Field(default=False, description="Did an error occur?")
    observed_time_saved_sec: Optional[float] = Field(
        None,
        ge=0.0,
        description="Actual time saved (if measurable)"
    )


class ResolveModeIn(BaseModel):
    """Request to determine execution mode for current context."""
    scope: ScopeKey
    model_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    model_safety: float = Field(default=0.85, ge=0.0, le=1.0)


# =====================================
# Response Schemas
# =====================================

class EventResponse(BaseModel):
    """Standard response for event logging."""
    ok: bool = True
    event_id: int
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExecutionResponse(BaseModel):
    """Response for execution creation."""
    ok: bool = True
    execution_id: int
    event_id: int
    reversible_until: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ScopeStatusResponse(BaseModel):
    """Current trust scope status."""
    scope: Dict[str, Any]
    stage: str
    trust: float
    metrics: Dict[str, float]
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResolvedModeResponse(BaseModel):
    """Execution mode resolution result."""
    ok: bool = True
    mode: str
    meta: Dict[str, Any]


# =====================================
# Helper Functions
# =====================================

def _enforce_tenant_isolation(scope_key: ScopeKey, user: UserContext) -> None:
    """
    Ensure user can only access their own tenant's data.

    Raises:
        HTTPException: 403 if tenant mismatch
    """
    # For demo, allow "default" tenant
    if scope_key.tenant_id == "default":
        return

    # For production, enforce strict isolation
    user_tenant = getattr(user, 'tenant_id', 'default')
    if scope_key.tenant_id != user_tenant:
        logger.warning(
            f"Tenant isolation violation: user={user.user_id}, "
            f"attempted_tenant={scope_key.tenant_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other tenant's data"
        )


def _get_request_id(request: Request) -> str:
    """Get or generate request ID for correlation."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


# =====================================
# Endpoints
# =====================================

@router.get("/scope", response_model=ScopeStatusResponse)
def get_scope_status(
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str] = None,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Get current trust scope status including stage, trust score, and metrics.

    Returns the latest computed trust metrics and automation stage.
    """
    # Enforce tenant isolation
    scope_key = ScopeKey(
        tenant_id=tenant_id,
        site_id=site_id,
        queue=queue,
        action_type=action_type,
        role_context=role_context
    )
    _enforce_tenant_isolation(scope_key, user)

    try:
        s = compute_and_update_scope(
            db, tenant_id, site_id, queue, action_type, role_context, AMEConfig()
        )

        return ScopeStatusResponse(
            scope={
                "tenant_id": s.tenant_id,
                "site_id": s.site_id,
                "queue": s.queue,
                "action_type": s.action_type,
                "role_context": s.role_context,
            },
            stage=s.stage,
            trust=float(s.trust_score),
            metrics={
                "reliability": float(s.reliability),
                "alignment": float(s.alignment),
                "safety_calibration": float(s.safety_calibration),
                "value_score": float(s.value_score),
                "override_rate": float(s.override_rate),
            },
            updated_at=s.updated_at,
        )
    except Exception as e:
        logger.error(f"Failed to get scope status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proposal", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
def ame_record_proposal(
    ev: ProposalEventIn,
    request: Request,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Record that AI proposed an automated action.

    This is the first step in the trust evolution cycle.
    """
    _enforce_tenant_isolation(ev.scope, user)
    request_id = _get_request_id(request)

    try:
        e = log_event(
            db,
            tenant_id=ev.scope.tenant_id,
            site_id=ev.scope.site_id,
            queue=ev.scope.queue,
            action_type=ev.scope.action_type,
            role_context=ev.scope.role_context,
            event_type=AMEEventType.PROPOSAL_CREATED.value,
            proposal_id=ev.proposal_id,
            predicted_confidence=ev.predicted_confidence,
            predicted_safety=ev.predicted_safety,
            predicted_time_saved_sec=ev.predicted_time_saved_sec,
            features=ev.features,
            context={**ev.context, "request_id": request_id},
        )

        # Audit log
        log_action(
            actor=user.user_id,
            role=user.role,
            action="ame.proposal_created",
            metadata={
                "event_id": e.id,
                "proposal_id": ev.proposal_id,
                "queue": ev.scope.queue,
                "action_type": ev.scope.action_type,
                "confidence": ev.predicted_confidence,
            },
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
        )

        return EventResponse(ok=True, event_id=e.id, timestamp=e.ts)

    except Exception as e:
        logger.error(f"Failed to record proposal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision", response_model=EventResponse)
def ame_record_decision(
    ev: DecisionEventIn,
    request: Request,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Record human decision on AI proposal (approve/reject/defer).

    This feeds the alignment metric in trust computation.
    """
    _enforce_tenant_isolation(ev.scope, user)
    request_id = _get_request_id(request)

    try:
        e = log_event(
            db,
            tenant_id=ev.scope.tenant_id,
            site_id=ev.scope.site_id,
            queue=ev.scope.queue,
            action_type=ev.scope.action_type,
            role_context=ev.scope.role_context,
            event_type=AMEEventType.PROPOSAL_DECIDED.value,
            proposal_id=ev.proposal_id,
            decision=ev.decision,
            decision_by=user.user_id,
            decision_reason=ev.decision_reason,
        )

        # Audit log
        log_action(
            actor=user.user_id,
            role=user.role,
            action=f"ame.proposal_{ev.decision}",
            metadata={
                "event_id": e.id,
                "proposal_id": ev.proposal_id,
                "reason": ev.decision_reason,
            },
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
        )

        return EventResponse(ok=True, event_id=e.id, timestamp=e.ts)

    except Exception as e:
        logger.error(f"Failed to record decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=ExecutionResponse)
def ame_record_execution(
    ev: ExecutionEventIn,
    request: Request,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Record that an action was executed.

    If guarded=true, creates a rollback window for human review.
    """
    _enforce_tenant_isolation(ev.scope, user)
    request_id = _get_request_id(request)

    try:
        # Create execution record (with rollback window if guarded)
        ex = create_execution(
            db,
            tenant_id=ev.scope.tenant_id,
            site_id=ev.scope.site_id,
            queue=ev.scope.queue,
            action_type=ev.scope.action_type,
            role_context=ev.scope.role_context,
            proposal_id=ev.proposal_id,
            before_state=ev.before_state,
            after_state=ev.after_state,
            guarded=ev.guarded,
        )

        # Log execution event
        e = log_event(
            db,
            tenant_id=ev.scope.tenant_id,
            site_id=ev.scope.site_id,
            queue=ev.scope.queue,
            action_type=ev.scope.action_type,
            role_context=ev.scope.role_context,
            event_type=AMEEventType.EXECUTED.value,
            proposal_id=ev.proposal_id,
            override=False,
        )

        # Audit log
        log_action(
            actor=user.user_id,
            role=user.role,
            action="ame.executed",
            metadata={
                "execution_id": ex.id,
                "event_id": e.id,
                "proposal_id": ev.proposal_id,
                "guarded": ev.guarded,
            },
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
        )

        return ExecutionResponse(
            ok=True,
            execution_id=ex.id,
            event_id=e.id,
            reversible_until=ex.reversible_until,
        )

    except Exception as e:
        logger.error(f"Failed to record execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/outcome", response_model=EventResponse)
def ame_record_outcome(
    ev: OutcomeEventIn,
    request: Request,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Record observed outcome of executed action.

    This feeds reliability and value metrics in trust computation.
    """
    _enforce_tenant_isolation(ev.scope, user)
    request_id = _get_request_id(request)

    try:
        e = log_event(
            db,
            tenant_id=ev.scope.tenant_id,
            site_id=ev.scope.site_id,
            queue=ev.scope.queue,
            action_type=ev.scope.action_type,
            role_context=ev.scope.role_context,
            event_type=AMEEventType.OUTCOME.value,
            proposal_id=ev.proposal_id,
            outcome_success=ev.outcome_success,
            observed_error=ev.observed_error,
            observed_time_saved_sec=ev.observed_time_saved_sec,
        )

        # Audit log
        log_action(
            actor=user.user_id,
            role=user.role,
            action="ame.outcome_recorded",
            metadata={
                "event_id": e.id,
                "proposal_id": ev.proposal_id,
                "success": ev.outcome_success,
                "error": ev.observed_error,
            },
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
        )

        return EventResponse(ok=True, event_id=e.id, timestamp=e.ts)

    except Exception as e:
        logger.error(f"Failed to record outcome: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/{execution_id}")
def ame_rollback(
    execution_id: int = Path(..., gt=0),
    reason: str = Body(..., embed=True, max_length=500),
    *,
    request: Request,
    user: UserContext = Depends(require_role("admin")),  # Only admins can rollback
    db: Session = Depends(get_db),
):
    """
    Rollback a previously executed action.

    **Requires admin role.**

    This creates a negative trust signal and may trigger stage downgrade.
    """
    request_id = _get_request_id(request)

    try:
        ex = rollback_execution(db, execution_id, reason=reason)

        # Audit log (critical action)
        log_action(
            actor=user.user_id,
            role=user.role,
            action="ame.rollback",
            metadata={
                "execution_id": execution_id,
                "proposal_id": ex.proposal_id,
                "reason": reason,
            },
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
            status="success",
        )

        return {
            "ok": True,
            "execution_id": ex.id,
            "rolled_back": ex.rolled_back,
            "rollback_reason": ex.rollback_reason,
            "rollback_at": ex.rollback_at.isoformat() if ex.rollback_at else None,
        }

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to rollback execution: {e}", exc_info=True)

        # Audit failed attempt
        log_action(
            actor=user.user_id,
            role=user.role,
            action="ame.rollback",
            metadata={"execution_id": execution_id, "error": str(e)},
            ip_address=request.client.host if request.client else None,
            request_id=request_id,
            status="failure",
        )

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve-mode", response_model=ResolvedModeResponse)
def ame_resolve_mode(
    req: ResolveModeIn,
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Determine execution mode for current context.

    Returns one of:
    - OBSERVE: AI observes only
    - PROPOSE: AI proposes, human decides
    - GUARDED_AUTO: AI acts, human can rollback
    - CONDITIONAL_AUTO: AI acts conditionally
    - FULL_AUTO: AI acts autonomously

    Mode depends on trust stage and anomaly detection.
    """
    _enforce_tenant_isolation(req.scope, user)

    try:
        mode, meta = resolve_execution_mode(
            db,
            tenant_id=req.scope.tenant_id,
            site_id=req.scope.site_id,
            queue=req.scope.queue,
            action_type=req.scope.action_type,
            role_context=req.scope.role_context,
            model_confidence=req.model_confidence,
            model_safety=req.model_safety,
        )

        return ResolvedModeResponse(ok=True, mode=mode, meta=meta)

    except Exception as e:
        logger.error(f"Failed to resolve mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
