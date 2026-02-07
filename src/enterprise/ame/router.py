"""
AME API - Adaptive Model Evolution REST endpoints
Manages trust scoring, stage progression, and execution tracking.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import desc

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
from .models import AMEEventType, AMETrustScope, AMEEvent
from .ml.trainer import AMETrainer
from .ml import clear_cache as _ml_clear_cache

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
        raise HTTPException(status_code=500, detail="Internal error")


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
        raise HTTPException(status_code=500, detail="Internal error")


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
        raise HTTPException(status_code=500, detail="Internal error")


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
        raise HTTPException(status_code=500, detail="Internal error")


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
        raise HTTPException(status_code=500, detail="Internal error")


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

    except ValueError:
        raise HTTPException(status_code=404, detail="Execution not found or not eligible for rollback")
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

        raise HTTPException(status_code=500, detail="Internal error")


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
        raise HTTPException(status_code=500, detail="Internal error")


# =====================================
# List / Dashboard Endpoints
# =====================================

@router.get("/scopes")
def list_scopes(
    tenant_id: str = Query("default"),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """List all trust scopes for a tenant with current metrics."""
    scopes = (
        db.query(AMETrustScope)
        .filter(AMETrustScope.tenant_id == tenant_id)
        .order_by(AMETrustScope.queue, AMETrustScope.action_type)
        .all()
    )
    return [
        {
            "id": s.id,
            "tenant_id": s.tenant_id,
            "site_id": s.site_id,
            "queue": s.queue,
            "action_type": s.action_type,
            "role_context": s.role_context,
            "stage": s.stage,
            "trust_score": float(s.trust_score) if s.trust_score else 0.0,
            "reliability": float(s.reliability) if s.reliability else 0.0,
            "alignment": float(s.alignment) if s.alignment else 0.0,
            "safety_calibration": float(s.safety_calibration) if s.safety_calibration else 0.0,
            "value_score": float(s.value_score) if s.value_score else 0.0,
            "override_rate": float(s.override_rate) if s.override_rate else 0.0,
            "total_proposals": s.total_proposals or 0,
            "approved_proposals": s.approved_proposals or 0,
            "successful_executions": s.successful_executions or 0,
            "stage_changed_at": s.stage_changed_at.isoformat() if s.stage_changed_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in scopes
    ]


@router.get("/events/recent")
def list_recent_events(
    tenant_id: str = Query("default"),
    limit: int = Query(50, ge=1, le=200),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """List recent AME events for a tenant, newest first."""
    events = (
        db.query(AMEEvent)
        .filter(AMEEvent.tenant_id == tenant_id, AMEEvent.deleted_at.is_(None))
        .order_by(desc(AMEEvent.ts))
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "queue": e.queue,
            "action_type": e.action_type,
            "proposal_id": e.proposal_id,
            "predicted_confidence": float(e.predicted_confidence) if e.predicted_confidence else None,
            "predicted_safety": float(e.predicted_safety) if e.predicted_safety else None,
            "decision": e.decision,
            "decision_by": e.decision_by,
            "decision_reason": e.decision_reason,
            "outcome_success": e.outcome_success,
            "observed_error": e.observed_error,
            "override": e.override,
            "ts": e.ts.isoformat() if e.ts else None,
        }
        for e in events
    ]


# =====================================
# ML Model Endpoints
# =====================================

@router.get("/ml/status")
def ame_ml_status(
    user: UserContext = Depends(require_auth),
):
    """Get status of all ML models: versions, training metrics, drift info."""
    trainer = AMETrainer()
    return trainer.get_status()


@router.post("/ml/retrain")
def ame_ml_retrain(
    user: UserContext = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """Retrain all ML models on current event data. Requires admin role."""
    trainer = AMETrainer()
    results = trainer.train_all(db)
    _ml_clear_cache()
    return {"ok": True, "results": results}


@router.get("/ml/retrain/check")
def ame_ml_retrain_check(
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Check if any models need retraining."""
    trainer = AMETrainer()
    return trainer.check_retrain_needed(db)


# =====================================
# AME Trust Dashboard (HTML)
# =====================================

@router.get("/dashboard", response_class=HTMLResponse)
def ame_dashboard_ui():
    """Visual dashboard showing live AME trust metrics and event timeline."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AME Trust Dashboard — Bladnir</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0b0f14;color:#e0e0e0}
    a{color:#60a5fa;text-decoration:none}
    a:hover{text-decoration:underline}
    header{padding:16px 24px;border-bottom:1px solid #1e2530;display:flex;align-items:center;gap:14px}
    header b{font-size:18px;color:#fff}
    .muted{color:#6b7280;font-size:13px}
    main{padding:24px;max-width:1200px;margin:0 auto}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:16px;margin-bottom:32px}
    .card{background:#111823;border:1px solid #1e2530;border-radius:12px;padding:16px}
    .card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
    .card-title{font-weight:600;font-size:15px;color:#fff}
    .badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
    .badge-observe{background:#374151;color:#9ca3af}
    .badge-propose{background:#1e3a5f;color:#60a5fa}
    .badge-guarded_auto{background:#4a3728;color:#fbbf24}
    .badge-conditional_auto{background:#4a2f1b;color:#fb923c}
    .badge-full_auto{background:#1a3a2a;color:#34d399}
    .trust-bar{height:8px;background:#1e2530;border-radius:4px;margin:8px 0;overflow:hidden}
    .trust-fill{height:100%;border-radius:4px;transition:width .5s}
    .trust-value{font-size:28px;font-weight:700;color:#fff}
    .metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;margin-top:12px}
    .metric{display:flex;justify-content:space-between;font-size:13px}
    .metric-label{color:#6b7280}
    .metric-value{color:#e0e0e0;font-weight:500}
    .metric-bar{height:4px;background:#1e2530;border-radius:2px;margin-top:2px;overflow:hidden}
    .metric-fill{height:100%;border-radius:2px}
    .stage-track{display:flex;gap:4px;margin:12px 0}
    .stage-dot{flex:1;height:6px;border-radius:3px;background:#1e2530;position:relative}
    .stage-dot.active{background:#34d399}
    .stage-dot.passed{background:#1a3a2a}
    .stage-labels{display:flex;justify-content:space-between;font-size:10px;color:#4b5563;margin-top:2px}
    .section-title{font-size:16px;font-weight:600;color:#fff;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
    .timeline{max-height:500px;overflow-y:auto}
    .ev-row{display:grid;grid-template-columns:130px 120px 1fr 80px;gap:8px;padding:8px 0;border-bottom:1px solid #1a1f2a;font-size:13px;align-items:center}
    .ev-type{font-weight:500}
    .ev-queue{color:#6b7280}
    .ev-detail{color:#9ca3af}
    .ev-time{color:#4b5563;text-align:right;font-size:12px}
    .type-proposal_created{color:#60a5fa}
    .type-proposal_decided{color:#a78bfa}
    .type-executed{color:#fbbf24}
    .type-outcome{color:#34d399}
    .type-rollback{color:#f87171}
    .type-anomaly{color:#fb923c}
    .type-stage_change{color:#e879f9}
    .empty{color:#4b5563;font-style:italic;padding:32px;text-align:center}
    .btn{padding:6px 14px;background:#1e2530;border:1px solid #2e3540;color:#e0e0e0;border-radius:6px;cursor:pointer;font-size:13px}
    .btn:hover{background:#2e3540}
    .counts{display:flex;gap:16px;margin-top:10px;flex-wrap:wrap}
    .count-item{font-size:12px;color:#6b7280}
    .count-item b{color:#e0e0e0}
    .nav-links{display:flex;gap:16px;font-size:13px}
  </style>
</head>
<body>
<header>
  <b>AME Trust Dashboard</b>
  <span class="muted">Adaptive Model Evolution — Live Trust Metrics</span>
  <div class="nav-links" style="margin-left:auto">
    <a href="/dashboard">Control Tower</a>
  </div>
  <button class="btn" onclick="loadAll()">Refresh</button>
  <span class="muted" id="lastUpdate"></span>
</header>

<main>
  <div class="section-title">
    <span>Trust Scopes</span>
    <span class="muted" id="scopeCount"></span>
  </div>
  <div class="grid" id="scopeGrid">
    <div class="empty">Loading scopes...</div>
  </div>

  <div class="section-title">
    <span>Recent Events</span>
    <span class="muted" id="eventCount"></span>
  </div>
  <div class="card">
    <div class="timeline" id="timeline">
      <div class="empty">Loading events...</div>
    </div>
  </div>
</main>

<script>
const STAGES=["observe","propose","guarded_auto","conditional_auto","full_auto"];
const STAGE_LABELS={observe:"Observe",propose:"Propose",guarded_auto:"Guarded Auto",conditional_auto:"Conditional Auto",full_auto:"Full Auto"};
const STAGE_SHORT={observe:"OBS",propose:"PROP",guarded_auto:"GRD",conditional_auto:"CND",full_auto:"FULL"};

function trustColor(t){
  if(t>=0.75)return"#34d399";
  if(t>=0.50)return"#fbbf24";
  if(t>=0.25)return"#fb923c";
  return"#f87171";
}

function metricColor(v){
  if(v>=0.7)return"#34d399";
  if(v>=0.4)return"#fbbf24";
  return"#f87171";
}

function renderScope(s){
  const idx=STAGES.indexOf(s.stage);
  const dots=STAGES.map((st,i)=>{
    let cls="stage-dot";
    if(i<idx)cls+=" passed";
    if(i===idx)cls+=" active";
    return'<div class="'+cls+'" title="'+STAGE_LABELS[st]+'"></div>';
  }).join("");
  const labels=STAGES.map(st=>'<span>'+STAGE_SHORT[st]+'</span>').join("");

  function mbar(val){
    const c=metricColor(val);
    return '<div class="metric-bar"><div class="metric-fill" style="width:'+(val*100)+'%;background:'+c+'"></div></div>';
  }

  return '<div class="card">'+
    '<div class="card-header">'+
      '<div>'+
        '<div class="card-title">'+s.queue+' / '+s.action_type+'</div>'+
        '<div class="muted">'+s.site_id+(s.role_context?' ('+s.role_context+')':'')+'</div>'+
      '</div>'+
      '<span class="badge badge-'+s.stage+'">'+(STAGE_LABELS[s.stage]||s.stage)+'</span>'+
    '</div>'+
    '<div style="display:flex;align-items:baseline;gap:8px">'+
      '<span class="trust-value">'+(s.trust_score*100).toFixed(1)+'%</span>'+
      '<span class="muted">trust score</span>'+
    '</div>'+
    '<div class="trust-bar"><div class="trust-fill" style="width:'+(s.trust_score*100)+'%;background:'+trustColor(s.trust_score)+'"></div></div>'+
    '<div class="stage-track">'+dots+'</div>'+
    '<div class="stage-labels">'+labels+'</div>'+
    '<div class="metrics">'+
      '<div class="metric"><span class="metric-label">Reliability</span><span class="metric-value">'+(s.reliability*100).toFixed(1)+'%</span></div>'+
      '<div class="metric"><span class="metric-label">Alignment</span><span class="metric-value">'+(s.alignment*100).toFixed(1)+'%</span></div>'+
      '<div class="metric"><span class="metric-label">Safety Cal.</span><span class="metric-value">'+(s.safety_calibration*100).toFixed(1)+'%</span></div>'+
      '<div class="metric"><span class="metric-label">Value Score</span><span class="metric-value">'+(s.value_score*100).toFixed(1)+'%</span></div>'+
      '<div class="metric"><span class="metric-label">Override Rate</span><span class="metric-value">'+(s.override_rate*100).toFixed(1)+'%</span></div>'+
    '</div>'+
    '<div class="counts">'+
      '<div class="count-item">Proposals: <b>'+s.total_proposals+'</b></div>'+
      '<div class="count-item">Approved: <b>'+s.approved_proposals+'</b></div>'+
      '<div class="count-item">Executions: <b>'+s.successful_executions+'</b></div>'+
    '</div>'+
  '</div>';
}

function renderEvent(e){
  const cls="type-"+(e.event_type||"");
  let parts=[];
  if(e.decision)parts.push(e.decision);
  if(e.decision_by)parts.push("by "+e.decision_by);
  if(e.decision_reason)parts.push('"'+e.decision_reason+'"');
  if(e.outcome_success!==null&&e.outcome_success!==undefined)parts.push(e.outcome_success?"success":"failed");
  if(e.predicted_confidence)parts.push("conf:"+e.predicted_confidence.toFixed(2));
  if(e.predicted_safety)parts.push("safety:"+e.predicted_safety.toFixed(2));
  if(e.override)parts.push("OVERRIDE");
  const detail=parts.join(" · ")||"—";
  const ts=e.ts?new Date(e.ts).toLocaleTimeString():"";
  return '<div class="ev-row">'+
    '<span class="ev-type '+cls+'">'+e.event_type+'</span>'+
    '<span class="ev-queue">'+e.queue+'</span>'+
    '<span class="ev-detail">'+detail+'</span>'+
    '<span class="ev-time">'+ts+'</span>'+
  '</div>';
}

async function loadAll(){
  try{
    const[scopes,events]=await Promise.all([
      fetch("/ame/scopes").then(r=>r.json()),
      fetch("/ame/events/recent?limit=50").then(r=>r.json()),
    ]);
    const grid=document.getElementById("scopeGrid");
    document.getElementById("scopeCount").textContent=scopes.length+" scope"+(scopes.length!==1?"s":"");
    if(!scopes.length){
      grid.innerHTML='<div class="empty">No trust scopes yet. Run through the Kroger demo to generate AME data.</div>';
    }else{
      grid.innerHTML=scopes.map(renderScope).join("");
    }
    const tl=document.getElementById("timeline");
    document.getElementById("eventCount").textContent=events.length+" event"+(events.length!==1?"s":"");
    if(!events.length){
      tl.innerHTML='<div class="empty">No events yet. Run through the Kroger demo to generate AME events.</div>';
    }else{
      tl.innerHTML=events.map(renderEvent).join("");
    }
    document.getElementById("lastUpdate").textContent="Updated "+new Date().toLocaleTimeString();
  }catch(err){
    console.error("Failed to load AME data:",err);
  }
}

loadAll();
setInterval(loadAll,10000);
</script>
</body>
</html>
    """
