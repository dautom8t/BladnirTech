"""
AME Trust System - Adaptive Model Evolution
Computes trust scores and manages stage progression for AI automation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc

from .models import AMETrustScope, AMEEvent, AMEExecution, AMEStage, AMEEventType

logger = logging.getLogger(__name__)


# =====================================
# Configuration
# =====================================

@dataclass
class AMEConfig:
    """Configuration for AME trust system."""

    # Recency decay (higher = faster decay, more weight to recent)
    lambda_decay: float = 0.03

    # Minimum evidence to progress
    min_observations: int = 25
    min_proposals: int = 20
    min_executions: int = 10

    # Stage promotion thresholds
    observe_to_propose: float = 0.40
    propose_to_guarded: float = 0.60
    guarded_to_conditional: float = 0.75
    conditional_to_full: float = 0.85

    # Stage downgrade threshold
    downgrade_below: float = 0.70
    downgrade_hysteresis_count: int = 3  # Must fail this many times before downgrade

    # Override safety limits
    max_override_rate_for_promotion: float = 0.10
    max_observed_error_rate: float = 0.05

    # Trust formula weights (must sum to 1.0)
    w_reliability: float = 0.35
    w_alignment: float = 0.25
    w_safety_calibration: float = 0.25
    w_value: float = 0.15

    # Override penalty (0.0-1.0 multiplier on override_rate)
    override_penalty_scale: float = 0.25

    # Guarded auto rollback window
    guarded_rollback_minutes: int = 15

    # Anomaly detection thresholds
    anomaly_min_confidence: float = 0.50
    anomaly_min_safety: float = 0.60
    anomaly_confidence_drop: float = 0.20
    anomaly_safety_drop: float = 0.20

    # Metric computation
    max_events_to_analyze: int = 500
    metric_cache_seconds: int = 300  # 5 minutes


# =====================================
# Utility Functions
# =====================================

def _now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def _weight(ts: datetime, now: datetime, lambda_decay: float) -> float:
    """
    Compute exponential decay weight based on event age.

    More recent events have higher weight.
    """
    if ts > now:
        logger.warning(f"Event timestamp {ts} is in the future!")
        return 1.0

    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return math.exp(-lambda_decay * age_days)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    return a / b if b > 0 else default


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, x))


def _safe_json_dumps(obj: Any) -> Optional[str]:
    """
    Safely serialize object to JSON.

    Falls back to string conversion for non-serializable types.
    """
    if obj is None:
        return None

    try:
        return json.dumps(obj, default=str)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": "serialization_failed", "type": str(type(obj))})


def _safe_json_loads(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """Safely deserialize JSON string."""
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception as e:
        logger.error(f"JSON deserialization failed: {e}")
        return None


# =====================================
# Core Functions
# =====================================

def _ensure_scope(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str],
) -> AMETrustScope:
    """
    Get or create trust scope.

    Handles race conditions with database unique constraint.
    """
    # Try to fetch existing
    scope = (
        db.query(AMETrustScope)
        .filter(
            AMETrustScope.tenant_id == tenant_id,
            AMETrustScope.site_id == site_id,
            AMETrustScope.queue == queue,
            AMETrustScope.action_type == action_type,
            AMETrustScope.role_context.is_(None) if role_context is None
                else AMETrustScope.role_context == role_context,
        )
        .first()
    )

    if scope:
        return scope

    # Create new (handle race condition)
    try:
        scope = AMETrustScope(
            tenant_id=tenant_id,
            site_id=site_id,
            queue=queue,
            action_type=action_type,
            role_context=role_context,
            stage=AMEStage.OBSERVE.value,
            trust_score=0.0,
            last_event_at=_now(),
            updated_at=_now(),
        )
        db.add(scope)
        db.flush()  # Get ID without committing
        logger.info(f"Created new trust scope: {scope}")
        return scope

    except IntegrityError:
        # Another request created it, rollback and fetch
        db.rollback()
        scope = (
            db.query(AMETrustScope)
            .filter(
                AMETrustScope.tenant_id == tenant_id,
                AMETrustScope.site_id == site_id,
                AMETrustScope.queue == queue,
                AMETrustScope.action_type == action_type,
                AMETrustScope.role_context.is_(None) if role_context is None
                    else AMETrustScope.role_context == role_context,
            )
            .first()
        )

        if not scope:
            raise RuntimeError("Failed to create or fetch trust scope")

        return scope


def log_event(
    db: Session,
    *,
    tenant_id: str = "default",
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str] = None,
    event_type: str,
    proposal_id: Optional[str] = None,
    predicted_confidence: Optional[float] = None,
    predicted_safety: Optional[float] = None,
    predicted_time_saved_sec: Optional[float] = None,
    decision: Optional[str] = None,
    decision_by: Optional[str] = None,
    decision_reason: Optional[str] = None,
    outcome_success: Optional[bool] = None,
    observed_error: Optional[bool] = None,
    observed_time_saved_sec: Optional[float] = None,
    override: Optional[bool] = None,
    override_by: Optional[str] = None,
    override_reason: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AMEEvent:
    """
    Log an AME event and update trust metrics.

    Uses single transaction for consistency.
    """
    # Validate inputs
    if predicted_confidence is not None and not (0.0 <= predicted_confidence <= 1.0):
        raise ValueError(f"predicted_confidence must be 0-1, got {predicted_confidence}")
    if predicted_safety is not None and not (0.0 <= predicted_safety <= 1.0):
        raise ValueError(f"predicted_safety must be 0-1, got {predicted_safety}")

    try:
        # Ensure scope exists
        scope = _ensure_scope(db, tenant_id, site_id, queue, action_type, role_context)

        # Create event
        ev = AMEEvent(
            tenant_id=tenant_id,
            site_id=site_id,
            queue=queue,
            action_type=action_type,
            role_context=role_context,
            event_type=event_type,
            proposal_id=proposal_id,
            predicted_confidence=predicted_confidence,
            predicted_safety=predicted_safety,
            predicted_time_saved_sec=predicted_time_saved_sec,
            decision=decision,
            decision_by=decision_by,
            decision_reason=decision_reason,
            outcome_success=outcome_success,
            observed_error=observed_error,
            observed_time_saved_sec=observed_time_saved_sec,
            override=override,
            override_by=override_by,
            override_reason=override_reason,
            features_json=_safe_json_dumps(features),
            context_json=_safe_json_dumps(context),
        )
        db.add(ev)
        db.flush()  # Get event ID

        # Update scope metrics (checks cache TTL)
        _recompute_scope_if_needed(db, scope)

        # Single commit
        db.commit()
        db.refresh(ev)

        logger.debug(f"Logged AME event: {event_type} for {queue}/{action_type}")
        return ev

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log AME event: {e}", exc_info=True)
        raise


def _recompute_scope_if_needed(db: Session, scope: AMETrustScope, cfg: Optional[AMEConfig] = None) -> None:
    """
    Recompute scope metrics if cache is stale.

    Prevents excessive recomputation on high-frequency events.
    """
    cfg = cfg or AMEConfig()
    now = _now()

    # Check cache TTL
    if scope.updated_at and (now - scope.updated_at).total_seconds() < cfg.metric_cache_seconds:
        logger.debug(f"Skipping recompute (cache fresh): {scope}")
        return

    # Recompute
    metrics = _compute_scope_metrics(
        db,
        scope.tenant_id,
        scope.site_id,
        scope.queue,
        scope.action_type,
        scope.role_context,
        cfg
    )

    # Check for stage transition
    old_stage = scope.stage
    new_stage = _determine_stage(old_stage, metrics, cfg)

    # Update scope
    scope.trust_score = float(metrics["trust"])
    scope.reliability = float(metrics["reliability"])
    scope.alignment = float(metrics["alignment"])
    scope.safety_calibration = float(metrics["safety_calibration"])
    scope.value_score = float(metrics["value_score"])
    scope.override_rate = float(metrics["override_rate"])
    scope.total_proposals = int(metrics["proposals"])
    scope.approved_proposals = int(metrics["approved"])
    scope.successful_executions = int(metrics["successes"])
    scope.stage = new_stage
    scope.last_event_at = now
    scope.updated_at = now

    # Log stage transition
    if new_stage != old_stage:
        logger.info(
            f"Stage transition: {old_stage} → {new_stage} "
            f"(tenant={scope.tenant_id}, site={scope.site_id}, queue={scope.queue}, "
            f"action={scope.action_type}, trust={scope.trust_score:.3f})"
        )

        scope.stage_changed_at = now
        scope.previous_stage = old_stage

    db.flush()


def _compute_scope_metrics(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str],
    cfg: AMEConfig,
) -> Dict[str, float]:
    """
    Compute trust metrics from event history.

    Returns dict with: trust, reliability, alignment, safety_calibration,
    value_score, override_rate, and evidence counts.
    """
    # Implementation continues with same logic but better structure...
    # (Keeping response concise - would include all the metric computation logic here)
    pass


# Rest of implementation...
