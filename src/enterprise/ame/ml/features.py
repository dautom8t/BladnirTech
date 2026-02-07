"""
AME ML Feature Engineering

Extracts features from AME events and trust scopes for model training and inference.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models import AMEEvent, AMETrustScope, AMEEventType

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Known categorical values for encoding
QUEUES = ["contact_manager", "inbound_comms", "data_entry", "pre_verification", "dispensing", "verification"]
ACTION_TYPES = ["advance_queue"]
STAGES = ["observe", "propose", "guarded_auto", "conditional_auto", "full_auto"]
INSURANCE_RESULTS = ["accepted", "rejected", "pa_required", "no_insurance", "unknown"]


def _encode_cat(value: str, categories: List[str]) -> int:
    """Encode categorical as integer. Unknown values map to len(categories)."""
    try:
        return categories.index(value)
    except ValueError:
        return len(categories)


def _parse_context(event: AMEEvent) -> Dict[str, Any]:
    """Parse context_json, handling str or dict."""
    ctx = event.context_json
    if ctx is None:
        return {}
    if isinstance(ctx, str):
        try:
            return json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return {}
    return ctx if isinstance(ctx, dict) else {}


# =====================================================================
# Decision Predictor Features
# =====================================================================

DECISION_FEATURE_NAMES = [
    "action_type", "approval_rate_7d", "events_last_hour", "hour_of_day",
    "insurance_result", "predicted_confidence", "predicted_safety",
    "predicted_time_saved", "queue", "rejection_streak",
    "scope_alignment", "scope_reliability", "scope_stage", "scope_trust",
]


def build_decision_features(
    *,
    queue: str,
    action_type: str,
    predicted_confidence: float,
    predicted_safety: float,
    predicted_time_saved: float = 0.0,
    scope_trust: float = 0.0,
    scope_stage: str = "observe",
    scope_reliability: float = 0.0,
    scope_alignment: float = 0.0,
    approval_rate_7d: float = 0.5,
    rejection_streak: int = 0,
    hour_of_day: int = 12,
    events_last_hour: int = 0,
    insurance_result: str = "unknown",
) -> Dict[str, float]:
    """Build feature dict for decision prediction."""
    return {
        "queue": _encode_cat(queue, QUEUES),
        "action_type": _encode_cat(action_type, ACTION_TYPES),
        "predicted_confidence": predicted_confidence,
        "predicted_safety": predicted_safety,
        "predicted_time_saved": predicted_time_saved,
        "scope_trust": scope_trust,
        "scope_stage": _encode_cat(scope_stage, STAGES),
        "scope_reliability": scope_reliability,
        "scope_alignment": scope_alignment,
        "approval_rate_7d": approval_rate_7d,
        "rejection_streak": rejection_streak,
        "hour_of_day": hour_of_day,
        "events_last_hour": events_last_hour,
        "insurance_result": _encode_cat(insurance_result, INSURANCE_RESULTS),
    }


# =====================================================================
# Outcome Predictor Features
# =====================================================================

OUTCOME_FEATURE_NAMES = [
    "action_type", "execution_count_today", "from_queue", "insurance_result",
    "queue", "recent_rollback_count", "recent_success_rate",
    "scope_override_rate", "scope_trust", "to_queue",
]


def build_outcome_features(
    *,
    queue: str,
    action_type: str,
    from_queue: str = "",
    to_queue: str = "",
    scope_trust: float = 0.0,
    scope_override_rate: float = 0.0,
    recent_success_rate: float = 0.5,
    recent_rollback_count: int = 0,
    execution_count_today: int = 0,
    insurance_result: str = "unknown",
) -> Dict[str, float]:
    """Build feature dict for outcome prediction."""
    return {
        "queue": _encode_cat(queue, QUEUES),
        "action_type": _encode_cat(action_type, ACTION_TYPES),
        "from_queue": _encode_cat(from_queue, QUEUES),
        "to_queue": _encode_cat(to_queue, QUEUES),
        "scope_trust": scope_trust,
        "scope_override_rate": scope_override_rate,
        "recent_success_rate": recent_success_rate,
        "recent_rollback_count": recent_rollback_count,
        "execution_count_today": execution_count_today,
        "insurance_result": _encode_cat(insurance_result, INSURANCE_RESULTS),
    }


# =====================================================================
# Anomaly Detector Features
# =====================================================================

ANOMALY_FEATURE_NAMES = [
    "avg_confidence", "avg_safety", "event_velocity", "override_rate",
    "proposal_count", "rejection_rate", "success_rate",
    "total_events", "unique_actors",
]


def build_anomaly_window(
    events: List[AMEEvent],
    window_start: datetime,
    window_end: datetime,
) -> Optional[Dict[str, float]]:
    """Compute aggregate features over a time window for anomaly detection."""
    window_events = [
        e for e in events
        if e.ts and window_start <= e.ts <= window_end
    ]

    if not window_events:
        return None

    duration_min = max(1.0, (window_end - window_start).total_seconds() / 60.0)

    proposals = [e for e in window_events if e.event_type == AMEEventType.PROPOSAL_CREATED.value]
    decisions = [e for e in window_events if e.event_type == AMEEventType.PROPOSAL_DECIDED.value]
    outcomes = [e for e in window_events if e.event_type == AMEEventType.OUTCOME.value]

    rejections = sum(1 for e in decisions if e.decision == "reject")
    overrides = sum(1 for e in window_events if e.override)
    successes = sum(1 for e in outcomes if e.outcome_success is True)

    confidences = [float(e.predicted_confidence) for e in window_events if e.predicted_confidence is not None]
    safeties = [float(e.predicted_safety) for e in window_events if e.predicted_safety is not None]

    actors = set()
    for e in window_events:
        if e.decision_by:
            actors.add(e.decision_by)

    return {
        "proposal_count": float(len(proposals)),
        "rejection_rate": rejections / len(decisions) if decisions else 0.0,
        "override_rate": overrides / len(outcomes) if outcomes else 0.0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.5,
        "avg_safety": sum(safeties) / len(safeties) if safeties else 0.5,
        "success_rate": successes / len(outcomes) if outcomes else 1.0,
        "unique_actors": float(len(actors)),
        "event_velocity": len(window_events) / duration_min,
        "total_events": float(len(window_events)),
    }


# =====================================================================
# Context computation helpers (query DB for derived features)
# =====================================================================

def compute_decision_context(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
) -> Dict[str, Any]:
    """Compute derived features for decision prediction from DB."""
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    one_hour_ago = now - timedelta(hours=1)

    # Recent decisions in last 7 days
    decisions = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.event_type == AMEEventType.PROPOSAL_DECIDED.value,
            AMEEvent.decision.in_(["approve", "reject"]),
            AMEEvent.ts >= seven_days_ago,
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(desc(AMEEvent.ts))
        .limit(100)
        .all()
    )

    approvals = sum(1 for e in decisions if e.decision == "approve")
    total = len(decisions)
    approval_rate = approvals / total if total > 0 else 0.5

    # Rejection streak
    streak = 0
    for e in decisions:
        if e.decision == "reject":
            streak += 1
        else:
            break

    # Events in last hour
    events_hour = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.ts >= one_hour_ago,
            AMEEvent.deleted_at.is_(None),
        )
        .count()
    )

    # Get scope
    scope = (
        db.query(AMETrustScope)
        .filter(
            AMETrustScope.tenant_id == tenant_id,
            AMETrustScope.site_id == site_id,
            AMETrustScope.queue == queue,
            AMETrustScope.action_type == action_type,
        )
        .first()
    )

    return {
        "approval_rate_7d": approval_rate,
        "rejection_streak": streak,
        "hour_of_day": now.hour,
        "events_last_hour": events_hour,
        "scope_trust": float(scope.trust_score) if scope else 0.0,
        "scope_stage": scope.stage if scope else "observe",
        "scope_reliability": float(scope.reliability) if scope else 0.0,
        "scope_alignment": float(scope.alignment) if scope else 0.0,
    }


def compute_outcome_context(
    db: Session,
    tenant_id: str,
    site_id: str,
    queue: str,
    action_type: str,
) -> Dict[str, Any]:
    """Compute derived features for outcome prediction from DB."""
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Recent outcomes
    outcomes = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.event_type == AMEEventType.OUTCOME.value,
            AMEEvent.outcome_success.isnot(None),
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(desc(AMEEvent.ts))
        .limit(20)
        .all()
    )

    successes = sum(1 for e in outcomes if e.outcome_success is True)
    total = len(outcomes)

    # Rollbacks in last 7 days
    rollbacks = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.event_type == AMEEventType.ROLLBACK.value,
            AMEEvent.ts >= seven_days_ago,
            AMEEvent.deleted_at.is_(None),
        )
        .count()
    )

    # Executions today
    exec_today = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.queue == queue,
            AMEEvent.event_type == AMEEventType.EXECUTED.value,
            AMEEvent.ts >= today_start,
            AMEEvent.deleted_at.is_(None),
        )
        .count()
    )

    # Scope
    scope = (
        db.query(AMETrustScope)
        .filter(
            AMETrustScope.tenant_id == tenant_id,
            AMETrustScope.site_id == site_id,
            AMETrustScope.queue == queue,
            AMETrustScope.action_type == action_type,
        )
        .first()
    )

    return {
        "recent_success_rate": successes / total if total > 0 else 0.5,
        "recent_rollback_count": rollbacks,
        "execution_count_today": exec_today,
        "scope_trust": float(scope.trust_score) if scope else 0.0,
        "scope_override_rate": float(scope.override_rate) if scope else 0.0,
    }


# =====================================================================
# Training data extraction
# =====================================================================

def get_training_data_decision(
    db: Session,
    tenant_id: str = "default",
    site_id: str = "dashboard_demo",
    limit: int = 500,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """Build training dataset for Decision Predictor. Returns (features, labels)."""
    decided = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.event_type == AMEEventType.PROPOSAL_DECIDED.value,
            AMEEvent.decision.in_(["approve", "reject"]),
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(desc(AMEEvent.ts))
        .limit(limit)
        .all()
    )

    if not decided:
        return [], []

    # Bulk-load context
    all_events = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(AMEEvent.ts)
        .all()
    )

    scopes = {
        s.queue: s
        for s in db.query(AMETrustScope).filter(
            AMETrustScope.tenant_id == tenant_id,
            AMETrustScope.site_id == site_id,
        ).all()
    }

    features_list = []
    labels = []

    for ev in decided:
        now = ev.ts or datetime.utcnow()
        scope = scopes.get(ev.queue)
        ctx = _parse_context(ev)

        # Compute contextual features from prior events
        prior = [e for e in all_events if e.ts and ev.ts and e.ts < ev.ts]
        seven_days_ago = now - timedelta(days=7)
        one_hour_ago = now - timedelta(hours=1)

        recent_decisions = [
            e for e in prior
            if e.event_type == AMEEventType.PROPOSAL_DECIDED.value
            and e.ts and e.ts >= seven_days_ago
            and e.queue == ev.queue
        ]
        approvals = sum(1 for e in recent_decisions if e.decision == "approve")
        total_d = len(recent_decisions)

        streak = 0
        for e in sorted(recent_decisions, key=lambda x: x.ts or now, reverse=True):
            if e.decision == "reject":
                streak += 1
            else:
                break

        events_hour = sum(
            1 for e in prior
            if e.ts and e.ts >= one_hour_ago and e.queue == ev.queue
        )

        features = build_decision_features(
            queue=ev.queue or "",
            action_type=ev.action_type or "",
            predicted_confidence=float(ev.predicted_confidence or 0.5),
            predicted_safety=float(ev.predicted_safety or 0.5),
            predicted_time_saved=float(ev.predicted_time_saved_sec or 0),
            scope_trust=float(scope.trust_score) if scope else 0.0,
            scope_stage=scope.stage if scope else "observe",
            scope_reliability=float(scope.reliability) if scope else 0.0,
            scope_alignment=float(scope.alignment) if scope else 0.0,
            approval_rate_7d=approvals / total_d if total_d > 0 else 0.5,
            rejection_streak=streak,
            hour_of_day=now.hour,
            events_last_hour=events_hour,
            insurance_result=ctx.get("insurance_result", "unknown"),
        )

        features_list.append(features)
        labels.append(1 if ev.decision == "approve" else 0)

    return features_list, labels


def get_training_data_outcome(
    db: Session,
    tenant_id: str = "default",
    site_id: str = "dashboard_demo",
    limit: int = 500,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """Build training dataset for Outcome Predictor. Returns (features, labels)."""
    outcomes = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.event_type == AMEEventType.OUTCOME.value,
            AMEEvent.outcome_success.isnot(None),
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(desc(AMEEvent.ts))
        .limit(limit)
        .all()
    )

    if not outcomes:
        return [], []

    all_events = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(AMEEvent.ts)
        .all()
    )

    scopes = {
        s.queue: s
        for s in db.query(AMETrustScope).filter(
            AMETrustScope.tenant_id == tenant_id,
            AMETrustScope.site_id == site_id,
        ).all()
    }

    features_list = []
    labels = []

    for ev in outcomes:
        now = ev.ts or datetime.utcnow()
        scope = scopes.get(ev.queue)
        ctx = _parse_context(ev)
        prior = [e for e in all_events if e.ts and ev.ts and e.ts < ev.ts and e.queue == ev.queue]

        # Parse transition
        transition = ctx.get("transition", "")
        from_q, to_q = "", ""
        if "\u2192" in transition:  # arrow character
            parts = transition.split("\u2192")
            from_q, to_q = parts[0].strip(), parts[1].strip()

        recent_outcomes = [e for e in prior if e.event_type == AMEEventType.OUTCOME.value][-20:]
        succ = sum(1 for e in recent_outcomes if e.outcome_success is True)
        total_o = len(recent_outcomes)

        seven_days_ago = now - timedelta(days=7)
        rollbacks = sum(
            1 for e in prior
            if e.event_type == AMEEventType.ROLLBACK.value and e.ts and e.ts >= seven_days_ago
        )

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        exec_today = sum(
            1 for e in prior if e.event_type == AMEEventType.EXECUTED.value and e.ts and e.ts >= today_start
        )

        features = build_outcome_features(
            queue=ev.queue or "",
            action_type=ev.action_type or "",
            from_queue=from_q,
            to_queue=to_q,
            scope_trust=float(scope.trust_score) if scope else 0.0,
            scope_override_rate=float(scope.override_rate) if scope else 0.0,
            recent_success_rate=succ / total_o if total_o > 0 else 0.5,
            recent_rollback_count=rollbacks,
            execution_count_today=exec_today,
            insurance_result=ctx.get("insurance_result", "unknown"),
        )

        features_list.append(features)
        labels.append(1 if ev.outcome_success else 0)

    return features_list, labels


def get_training_data_anomaly(
    db: Session,
    tenant_id: str = "default",
    site_id: str = "dashboard_demo",
    window_hours: int = 1,
) -> List[Dict[str, float]]:
    """Build training dataset for Anomaly Detector (unsupervised, no labels)."""
    events = (
        db.query(AMEEvent)
        .filter(
            AMEEvent.tenant_id == tenant_id,
            AMEEvent.site_id == site_id,
            AMEEvent.deleted_at.is_(None),
        )
        .order_by(AMEEvent.ts)
        .all()
    )

    if not events or not events[0].ts:
        return []

    start = events[0].ts
    end = events[-1].ts or datetime.utcnow()

    windows = []
    current = start
    while current < end:
        window_end = current + timedelta(hours=window_hours)
        feats = build_anomaly_window(events, current, window_end)
        if feats:
            windows.append(feats)
        current = window_end

    return windows


def features_to_array(features_list: List[Dict[str, float]], feature_names: List[str]):
    """Convert feature dicts to numpy array with consistent column order."""
    if not HAS_NUMPY or not features_list:
        return None
    return np.array(
        [[f.get(k, 0.0) for k in feature_names] for f in features_list],
        dtype=np.float64,
    )
