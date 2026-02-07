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

    # Minimum evidence to progress (low for demo; raise for production)
    min_observations: int = 6
    min_proposals: int = 3
    min_executions: int = 3

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
    metric_cache_seconds: int = 0  # Recompute on every event (demo-safe; raise for production)


# =====================================
# Utility Functions
# =====================================

def _now() -> datetime:
    """Return current UTC datetime (naive — compatible with SQLite which strips tzinfo)."""
    return datetime.utcnow()


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

    # Always recompute if scope has never had metrics computed (bootstrapping)
    never_computed = (
        float(scope.trust_score) == 0.0
        and scope.total_proposals == 0
        and scope.successful_executions == 0
    )

    # Check cache TTL — skip for never-computed scopes so metrics update immediately
    if not never_computed and scope.updated_at and (now - scope.updated_at).total_seconds() < cfg.metric_cache_seconds:
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
    Compute trust metrics from event history using exponential decay weighting.

    Returns dict with: trust, reliability, alignment, safety_calibration,
    value_score, override_rate, and evidence counts.
    """
    now = _now()

    # Fetch recent events for this scope
    q = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.action_type == action_type,
            AMEEvent.deleted_at.is_(None),
        )
    )
    # role_context filter MUST come before order_by/limit
    if role_context is not None:
        q = q.filter(AMEEvent.role_context == role_context)
    else:
        q = q.filter(AMEEvent.role_context.is_(None))

    events = q.order_by(desc(AMEEvent.ts)).limit(cfg.max_events_to_analyze).all()

    if not events:
        return {
            "trust": 0.0, "reliability": 0.0, "alignment": 0.0,
            "safety_calibration": 0.0, "value_score": 0.0, "override_rate": 0.0,
            "proposals": 0, "approved": 0, "rejected": 0,
            "successes": 0, "failures": 0, "overrides": 0,
            "observations": len(events),
        }

    # --- Counters ---
    proposals = 0
    approved = 0
    rejected = 0
    successes = 0
    failures = 0
    overrides = 0

    # --- Weighted accumulators ---
    w_correct_sum = 0.0
    w_correct_total = 0.0

    w_align_sum = 0.0
    w_align_total = 0.0

    w_safety_sum = 0.0
    w_safety_total = 0.0

    w_value_sum = 0.0
    w_value_total = 0.0

    w_override_sum = 0.0
    w_override_total = 0.0

    for ev in events:
        w = _weight(ev.ts, now, cfg.lambda_decay)

        if ev.event_type == AMEEventType.PROPOSAL_CREATED.value:
            proposals += 1

        elif ev.event_type == AMEEventType.PROPOSAL_DECIDED.value:
            if ev.decision == "approve":
                approved += 1
                w_align_sum += w * 1.0
            elif ev.decision == "reject":
                rejected += 1
                w_align_sum += w * 0.0
            w_align_total += w

        elif ev.event_type == AMEEventType.OUTCOME.value:
            if ev.outcome_success is True:
                successes += 1
                w_correct_sum += w * 1.0
            elif ev.outcome_success is False:
                failures += 1
                w_correct_sum += w * 0.0
            w_correct_total += w

            # Safety calibration: predicted_safety vs actual outcome
            if ev.predicted_safety is not None and ev.outcome_success is not None:
                actual = 1.0 if ev.outcome_success and not ev.observed_error else 0.0
                predicted = float(ev.predicted_safety)
                calibration = 1.0 - abs(predicted - actual)
                w_safety_sum += w * calibration
                w_safety_total += w

            # Value: observed time saved vs predicted
            if ev.observed_time_saved_sec is not None and ev.predicted_time_saved_sec is not None:
                predicted_t = float(ev.predicted_time_saved_sec)
                observed_t = float(ev.observed_time_saved_sec)
                if predicted_t > 0:
                    ratio = min(observed_t / predicted_t, 1.5)  # Cap at 150%
                    w_value_sum += w * _clamp01(ratio)
                    w_value_total += w

            # Override tracking
            if ev.override:
                overrides += 1
                w_override_sum += w * 1.0
            w_override_total += w

    # --- Compute final metrics ---
    reliability = _clamp01(_safe_div(w_correct_sum, w_correct_total))
    alignment = _clamp01(_safe_div(w_align_sum, w_align_total))
    safety_calibration = _clamp01(_safe_div(w_safety_sum, w_safety_total, default=0.5))
    value_score = _clamp01(_safe_div(w_value_sum, w_value_total, default=0.5))
    override_rate = _clamp01(_safe_div(w_override_sum, w_override_total))

    # Composite trust score
    raw_trust = (
        cfg.w_reliability * reliability
        + cfg.w_alignment * alignment
        + cfg.w_safety_calibration * safety_calibration
        + cfg.w_value * value_score
    )
    # Apply override penalty
    trust = _clamp01(raw_trust - cfg.override_penalty_scale * override_rate)

    return {
        "trust": trust,
        "reliability": reliability,
        "alignment": alignment,
        "safety_calibration": safety_calibration,
        "value_score": value_score,
        "override_rate": override_rate,
        "proposals": proposals,
        "approved": approved,
        "rejected": rejected,
        "successes": successes,
        "failures": failures,
        "overrides": overrides,
        "observations": len(events),
    }


def _determine_stage(
    current_stage: str,
    metrics: Dict[str, float],
    cfg: AMEConfig,
) -> str:
    """
    Determine the appropriate stage based on current metrics.

    Supports both promotion and demotion with hysteresis.
    """
    trust = metrics["trust"]
    observations = metrics["observations"]
    proposals = metrics["proposals"]
    successes = metrics["successes"]
    override_rate = metrics["override_rate"]

    # Safety guard: too many overrides blocks promotion
    override_safe = override_rate <= cfg.max_override_rate_for_promotion

    stage_order = [
        AMEStage.OBSERVE.value,
        AMEStage.PROPOSE.value,
        AMEStage.GUARDED_AUTO.value,
        AMEStage.CONDITIONAL_AUTO.value,
        AMEStage.FULL_AUTO.value,
    ]

    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0

    # --- Promotion logic ---
    target_idx = current_idx

    if (current_idx == 0
            and trust >= cfg.observe_to_propose
            and observations >= cfg.min_observations
            and override_safe):
        target_idx = 1

    if (current_idx <= 1
            and trust >= cfg.propose_to_guarded
            and proposals >= cfg.min_proposals
            and override_safe):
        target_idx = 2

    if (current_idx <= 2
            and trust >= cfg.guarded_to_conditional
            and successes >= cfg.min_executions
            and override_safe):
        target_idx = 3

    if (current_idx <= 3
            and trust >= cfg.conditional_to_full
            and successes >= cfg.min_executions * 2
            and override_safe):
        target_idx = 4

    # --- Demotion logic ---
    if current_idx >= 2 and trust < cfg.downgrade_below:
        # Drop one stage
        target_idx = min(target_idx, current_idx - 1)

    # Only move one stage at a time
    if target_idx > current_idx:
        target_idx = current_idx + 1
    elif target_idx < current_idx:
        target_idx = current_idx - 1

    return stage_order[target_idx]


# =====================================
# Public API (imported by router)
# =====================================

def get_scope(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str] = None,
) -> Optional[AMETrustScope]:
    """Fetch a trust scope without creating it."""
    return (
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


def compute_and_update_scope(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str],
    cfg: AMEConfig,
) -> AMETrustScope:
    """Get-or-create scope and force a metrics recompute."""
    scope = _ensure_scope(db, tenant_id, site_id, queue, action_type, role_context)

    metrics = _compute_scope_metrics(
        db, tenant_id, site_id, queue, action_type, role_context, cfg
    )

    old_stage = scope.stage
    new_stage = _determine_stage(old_stage, metrics, cfg)
    now = _now()

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

    if new_stage != old_stage:
        scope.stage_changed_at = now
        scope.previous_stage = old_stage

    db.commit()
    db.refresh(scope)
    return scope


def create_execution(
    db: Session,
    *,
    tenant_id: str = "default",
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str] = None,
    proposal_id: Optional[str] = None,
    before_state: Optional[Dict[str, Any]] = None,
    after_state: Optional[Dict[str, Any]] = None,
    guarded: bool = True,
) -> AMEExecution:
    """
    Create an execution record, optionally with a rollback window.

    If guarded=True, sets reversible_until to now + guarded_rollback_minutes.
    """
    cfg = AMEConfig()
    now = _now()

    reversible_until = None
    if guarded:
        reversible_until = now + timedelta(minutes=cfg.guarded_rollback_minutes)

    ex = AMEExecution(
        tenant_id=tenant_id,
        site_id=site_id,
        queue=queue,
        action_type=action_type,
        role_context=role_context,
        proposal_id=proposal_id,
        before_state_json=before_state,
        after_state_json=after_state,
        reversible_until=reversible_until,
        executed_by="system",
    )
    db.add(ex)
    db.flush()

    logger.info(
        f"Created execution {ex.id} for {queue}/{action_type} "
        f"(guarded={guarded}, reversible_until={reversible_until})"
    )
    return ex


def rollback_execution(
    db: Session,
    execution_id: int,
    *,
    reason: str = "",
    rolled_back_by: str = "admin",
) -> AMEExecution:
    """
    Roll back an execution if it's still within the reversibility window.

    Raises ValueError if execution not found or already rolled back.
    """
    ex = db.query(AMEExecution).filter(AMEExecution.id == execution_id).first()
    if not ex:
        raise ValueError(f"Execution {execution_id} not found")

    if ex.rolled_back:
        raise ValueError(f"Execution {execution_id} already rolled back")

    now = _now()

    if ex.reversible_until and now > ex.reversible_until:
        logger.warning(
            f"Rollback attempted after window closed for execution {execution_id}"
        )
        # Allow rollback but log the warning — admins may need to force it

    ex.rolled_back = True
    ex.rollback_at = now
    ex.rollback_by = rolled_back_by
    ex.rollback_reason = reason
    ex.rollback_success = True

    # Log rollback event for trust computation
    log_event(
        db,
        tenant_id=ex.tenant_id,
        site_id=ex.site_id,
        queue=ex.queue,
        action_type=ex.action_type,
        role_context=ex.role_context,
        event_type=AMEEventType.ROLLBACK.value,
        proposal_id=ex.proposal_id,
        override=True,
        override_by=rolled_back_by,
        override_reason=reason,
    )

    logger.info(f"Rolled back execution {execution_id}: {reason}")
    return ex


def resolve_execution_mode(
    db: Session,
    *,
    tenant_id: str = "default",
    site_id: str,
    queue: str,
    action_type: str,
    role_context: Optional[str] = None,
    model_confidence: float = 0.75,
    model_safety: float = 0.85,
) -> Tuple[str, Dict[str, Any]]:
    """
    Determine execution mode based on current trust stage and anomaly detection.

    Returns:
        Tuple of (mode_string, metadata_dict)
    """
    cfg = AMEConfig()
    scope = get_scope(db, tenant_id, site_id, queue, action_type, role_context)

    if scope is None:
        return AMEStage.OBSERVE.value, {
            "reason": "no_scope",
            "trust": 0.0,
            "stage": AMEStage.OBSERVE.value,
        }

    stage = scope.stage
    trust = float(scope.trust_score)

    # Anomaly detection: if confidence or safety drops below thresholds,
    # downgrade to PROPOSE regardless of stage
    anomaly = False
    anomaly_reasons: List[str] = []

    if model_confidence < cfg.anomaly_min_confidence:
        anomaly = True
        anomaly_reasons.append(f"low_confidence({model_confidence:.2f})")

    if model_safety < cfg.anomaly_min_safety:
        anomaly = True
        anomaly_reasons.append(f"low_safety({model_safety:.2f})")

    if anomaly and stage in (
        AMEStage.GUARDED_AUTO.value,
        AMEStage.CONDITIONAL_AUTO.value,
        AMEStage.FULL_AUTO.value,
    ):
        return AMEStage.PROPOSE.value, {
            "reason": "anomaly_detected",
            "anomaly_reasons": anomaly_reasons,
            "original_stage": stage,
            "trust": trust,
            "stage": AMEStage.PROPOSE.value,
        }

    return stage, {
        "reason": "normal",
        "trust": trust,
        "stage": stage,
        "model_confidence": model_confidence,
        "model_safety": model_safety,
    }
