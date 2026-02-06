"""
AME (Adaptive Model Evolution) Trust System
Tracks trust scores, stage progression, and event history for AI automation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    Index, UniqueConstraint, Numeric, ForeignKey, CheckConstraint
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import declarative_base, relationship, validates

Base = declarative_base()


# =====================================
# Enums for Type Safety
# =====================================

class AMEStage(str, Enum):
    """Trust/automation maturity stages."""
    OBSERVE = "observe"              # AI observes, humans decide
    PROPOSE = "propose"              # AI proposes, humans decide
    GUARDED_AUTO = "guarded_auto"    # AI acts, humans can revert
    CONDITIONAL_AUTO = "conditional_auto"  # AI acts under conditions
    FULL_AUTO = "full_auto"          # AI acts autonomously


class AMEEventType(str, Enum):
    """Event types in the trust evolution lifecycle."""
    PROPOSAL_CREATED = "proposal_created"
    PROPOSAL_DECIDED = "proposal_decided"
    EXECUTED = "executed"
    OUTCOME = "outcome"
    ROLLBACK = "rollback"
    ANOMALY = "anomaly"
    STAGE_CHANGE = "stage_change"


class AMEDecision(str, Enum):
    """Human decision on AI proposal."""
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


# =====================================
# Main Models
# =====================================

class AMETrustScope(Base):
    """
    Trust scope for a specific automation context.

    One record per (tenant, site, queue, action, role) combination.
    Tracks the current trust stage and metrics for AI automation maturity.

    Trust Metrics:
    - trust_score: Overall trust (0.0-1.0)
    - reliability: Consistency of AI predictions (0.0-1.0)
    - alignment: How well AI aligns with human decisions (0.0-1.0)
    - safety_calibration: AI's safety assessment accuracy (0.0-1.0)
    - value_score: Business value delivered (0.0-1.0)
    - override_rate: Frequency of human overrides (0.0-1.0)
    """
    __tablename__ = "ame_trust_scopes"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Scope identifiers
    tenant_id = Column(String(120), default="default", nullable=False, index=True)
    site_id = Column(String(120), nullable=False, index=True)
    queue = Column(String(120), nullable=False, index=True)
    action_type = Column(String(120), nullable=False, index=True)
    role_context = Column(String(120), nullable=True)

    # State machine stage
    stage = Column(
        String(32),
        default=AMEStage.OBSERVE.value,
        nullable=False,
        comment="Current automation maturity stage"
    )

    # Stage change audit
    stage_changed_at = Column(DateTime, nullable=True)
    stage_changed_by = Column(String(100), nullable=True)
    stage_change_reason = Column(String(500), nullable=True)
    previous_stage = Column(String(32), nullable=True)

    # Trust metrics (using Numeric for precision)
    trust_score = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="Overall trust score (0.0-1.0)"
    )
    reliability = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="AI prediction consistency"
    )
    alignment = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="AI-human decision alignment"
    )
    safety_calibration = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="Safety assessment accuracy"
    )
    value_score = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="Business value delivered"
    )
    override_rate = Column(
        Numeric(precision=5, scale=4),
        default=0.0,
        nullable=False,
        comment="Human override frequency"
    )

    # Counts and stats
    total_proposals = Column(Integer, default=0, nullable=False)
    approved_proposals = Column(Integer, default=0, nullable=False)
    rejected_proposals = Column(Integer, default=0, nullable=False)
    successful_executions = Column(Integer, default=0, nullable=False)
    failed_executions = Column(Integer, default=0, nullable=False)
    rollbacks = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    last_event_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "site_id", "queue", "action_type", "role_context",
            name="uq_ame_scope"
        ),
        Index("ix_ame_scope_lookup", "tenant_id", "site_id", "queue", "action_type"),
        Index("ix_ame_scope_stage", "stage"),
        CheckConstraint("trust_score >= 0 AND trust_score <= 1", name="ck_trust_score"),
        CheckConstraint("reliability >= 0 AND reliability <= 1", name="ck_reliability"),
    )

    @validates('stage')
    def validate_stage(self, key, value):
        """Validate stage is a known value."""
        valid_stages = [s.value for s in AMEStage]
        if value not in valid_stages:
            raise ValueError(f"Invalid stage: {value}. Must be one of {valid_stages}")
        return value

    def __repr__(self):
        return (
            f"<AMETrustScope(id={self.id}, tenant={self.tenant_id}, "
            f"site={self.site_id}, queue={self.queue}, action={self.action_type}, "
            f"stage={self.stage}, trust={self.trust_score})>"
        )


class AMEEvent(Base):
    """
    Immutable event log for trust evolution.

    Records every proposal, decision, execution, and outcome.
    Forms the training data spine for trust metric computation.

    Event Flow:
    1. proposal_created: AI suggests action
    2. proposal_decided: Human approves/rejects
    3. executed: Action performed
    4. outcome: Result observed
    5. rollback: Action reverted (if needed)
    """
    __tablename__ = "ame_events"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Timestamp (timezone-aware)
    ts = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    # Scope identifiers
    tenant_id = Column(String(120), default="default", nullable=False, index=True)
    site_id = Column(String(120), nullable=False, index=True)
    queue = Column(String(120), nullable=False, index=True)
    action_type = Column(String(120), nullable=False, index=True)
    role_context = Column(String(120), nullable=True)

    # Event classification
    event_type = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Type of event (proposal_created, decided, executed, etc.)"
    )
    proposal_id = Column(String(64), nullable=True, index=True)

    # AI model outputs at decision time
    predicted_confidence = Column(Numeric(precision=5, scale=4), nullable=True)
    predicted_safety = Column(Numeric(precision=5, scale=4), nullable=True)
    predicted_time_saved_sec = Column(Integer, nullable=True)

    # Decision tracking
    decision = Column(String(32), nullable=True)  # approve, reject, defer
    decision_by = Column(String(100), nullable=True)  # Who decided
    decision_reason = Column(String(500), nullable=True)

    # Outcome tracking
    outcome_success = Column(Boolean, nullable=True)
    observed_error = Column(Boolean, nullable=True)
    error_type = Column(String(100), nullable=True)
    observed_time_saved_sec = Column(Integer, nullable=True)
    observed_cost_impact = Column(Numeric(precision=10, scale=2), nullable=True)

    # Override tracking
    override = Column(Boolean, default=False, nullable=False)
    override_by = Column(String(100), nullable=True)
    override_reason = Column(String(500), nullable=True)

    # Feature snapshot (JSON for flexibility)
    features_json = Column(JSON, nullable=True, comment="Input features at decision time")
    context_json = Column(JSON, nullable=True, comment="Context: staff load, SLA, backlog")
    metadata_json = Column(JSON, nullable=True, comment="Additional metadata")

    # Soft delete (events are immutable, but may need to hide bad data)
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String(100), nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("ix_ame_events_scope_ts", "tenant_id", "site_id", "queue", "action_type", "ts"),
        Index("ix_ame_events_proposal", "proposal_id"),
        Index("ix_ame_events_type", "event_type"),
        Index("ix_ame_events_decision", "decision"),
    )

    @validates('event_type')
    def validate_event_type(self, key, value):
        """Validate event type is known."""
        valid_types = [t.value for t in AMEEventType]
        if value not in valid_types:
            raise ValueError(f"Invalid event_type: {value}")
        return value

    def __repr__(self):
        return (
            f"<AMEEvent(id={self.id}, type={self.event_type}, "
            f"proposal={self.proposal_id}, ts={self.ts})>"
        )


class AMEExecution(Base):
    """
    Tracks reversible execution windows.

    Records the before/after state of automated actions with ability to rollback.
    Critical for guarded_auto and conditional_auto stages.
    """
    __tablename__ = "ame_executions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Timestamp
    ts = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    # Scope identifiers
    tenant_id = Column(String(120), default="default", nullable=False, index=True)
    site_id = Column(String(120), nullable=False, index=True)
    queue = Column(String(120), nullable=False, index=True)
    action_type = Column(String(120), nullable=False, index=True)
    role_context = Column(String(120), nullable=True)

    # Proposal reference
    proposal_id = Column(String(64), nullable=True, index=True)

    # State snapshots (JSON for flexibility)
    before_state_json = Column(JSON, nullable=True, comment="State before execution")
    after_state_json = Column(JSON, nullable=True, comment="State after execution")
    diff_json = Column(JSON, nullable=True, comment="Computed diff for quick review")

    # Reversibility window
    reversible_until = Column(
        DateTime,
        nullable=True,
        comment="Deadline for rollback"
    )

    # Rollback tracking
    rolled_back = Column(Boolean, default=False, nullable=False, index=True)
    rollback_at = Column(DateTime, nullable=True)
    rollback_by = Column(String(100), nullable=True)
    rollback_reason = Column(String(500), nullable=True)
    rollback_success = Column(Boolean, nullable=True)

    # Execution metadata
    executed_by = Column(String(100), nullable=True)
    execution_duration_ms = Column(Integer, nullable=True)

    # Indexes
    __table_args__ = (
        Index("ix_ame_exec_scope_ts", "tenant_id", "site_id", "queue", "action_type", "ts"),
        Index("ix_ame_exec_proposal", "proposal_id"),
        Index("ix_ame_exec_reversible", "reversible_until"),
        Index("ix_ame_exec_rolled_back", "rolled_back"),
    )

    def __repr__(self):
        return (
            f"<AMEExecution(id={self.id}, proposal={self.proposal_id}, "
            f"rolled_back={self.rolled_back}, ts={self.ts})>"
        )


# =====================================
# Helper Functions
# =====================================

def get_scope_key(tenant_id: str, site_id: str, queue: str, action_type: str, role_context: Optional[str] = None) -> str:
    """Generate unique key for a trust scope."""
    parts = [tenant_id, site_id, queue, action_type]
    if role_context:
        parts.append(role_context)
    return ":".join(parts)
