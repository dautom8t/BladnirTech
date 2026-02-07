"""Unit tests for AME trust calculations and stage progression."""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timedelta

from src.enterprise.ame.models import AMEStage, AMEEventType, AMETrustScope, AMEEvent
from src.enterprise.ame.service import (
    AMEConfig,
    log_event,
    get_scope,
    compute_and_update_scope,
    create_execution,
    rollback_execution,
    resolve_execution_mode,
    _weight,
    _safe_div,
    _clamp01,
    _determine_stage,
)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_weight_recent_event(self):
        now = datetime.utcnow()
        assert _weight(now, now, 0.03) == pytest.approx(1.0)

    def test_weight_old_event(self):
        now = datetime.utcnow()
        old = now - timedelta(days=30)
        w = _weight(old, now, 0.03)
        assert w < 1.0
        assert w == pytest.approx(math.exp(-0.03 * 30), rel=1e-4)

    def test_weight_future_event_returns_1(self):
        now = datetime.utcnow()
        future = now + timedelta(hours=1)
        assert _weight(future, now, 0.03) == 1.0

    def test_safe_div_normal(self):
        assert _safe_div(10, 2) == 5.0

    def test_safe_div_zero_denominator(self):
        assert _safe_div(10, 0) == 0.0
        assert _safe_div(10, 0, default=0.5) == 0.5

    def test_clamp01(self):
        assert _clamp01(0.5) == 0.5
        assert _clamp01(-0.1) == 0.0
        assert _clamp01(1.5) == 1.0


# ---------------------------------------------------------------------------
# Stage determination
# ---------------------------------------------------------------------------

class TestStageProgression:

    def _metrics(self, **overrides):
        """Base metrics with overridable fields."""
        base = {
            "trust": 0.0,
            "reliability": 0.0,
            "alignment": 0.0,
            "safety_calibration": 0.0,
            "value_score": 0.0,
            "override_rate": 0.0,
            "proposals": 0,
            "approved": 0,
            "rejected": 0,
            "successes": 0,
            "failures": 0,
            "overrides": 0,
            "observations": 0,
        }
        base.update(overrides)
        return base

    def test_observe_stays_without_evidence(self):
        cfg = AMEConfig()
        stage = _determine_stage("observe", self._metrics(), cfg)
        assert stage == "observe"

    def test_observe_to_propose(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.45, observations=10)
        stage = _determine_stage("observe", m, cfg)
        assert stage == "propose"

    def test_propose_to_guarded(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.65, observations=10, proposals=5)
        stage = _determine_stage("propose", m, cfg)
        assert stage == "guarded_auto"

    def test_guarded_to_conditional(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.80, observations=10, proposals=5, successes=5)
        stage = _determine_stage("guarded_auto", m, cfg)
        assert stage == "conditional_auto"

    def test_conditional_to_full(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.90, observations=10, proposals=5, successes=10)
        stage = _determine_stage("conditional_auto", m, cfg)
        assert stage == "full_auto"

    def test_only_one_stage_at_a_time(self):
        """Even with very high trust, can only jump one stage per evaluation."""
        cfg = AMEConfig()
        m = self._metrics(trust=0.95, observations=20, proposals=10, successes=15)
        stage = _determine_stage("observe", m, cfg)
        assert stage == "propose"  # one step, not straight to full_auto

    def test_demotion_on_low_trust(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.50, observations=10, proposals=5, successes=3)
        stage = _determine_stage("guarded_auto", m, cfg)
        assert stage == "propose"

    def test_override_blocks_promotion(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.50, observations=10, override_rate=0.20)
        stage = _determine_stage("observe", m, cfg)
        assert stage == "observe"  # blocked by high override rate

    def test_full_auto_stays_with_good_metrics(self):
        cfg = AMEConfig()
        m = self._metrics(trust=0.90, observations=20, proposals=10, successes=10)
        stage = _determine_stage("full_auto", m, cfg)
        assert stage == "full_auto"


# ---------------------------------------------------------------------------
# Core service functions
# ---------------------------------------------------------------------------

class TestLogEvent:

    def test_log_proposal_event(self, db_session):
        ev = log_event(
            db_session,
            site_id="test_site",
            queue="data_entry",
            action_type="advance_queue",
            event_type=AMEEventType.PROPOSAL_CREATED.value,
            proposal_id="P-test-01",
            predicted_confidence=0.85,
            predicted_safety=0.90,
        )
        assert ev.id is not None
        assert ev.event_type == "proposal_created"
        assert float(ev.predicted_confidence) == pytest.approx(0.85)

    def test_log_decision_event(self, db_session):
        ev = log_event(
            db_session,
            site_id="test_site",
            queue="data_entry",
            action_type="advance_queue",
            event_type=AMEEventType.PROPOSAL_DECIDED.value,
            proposal_id="P-test-01",
            decision="approve",
            decision_by="pharmacist",
        )
        assert ev.decision == "approve"

    def test_log_outcome_event(self, db_session):
        ev = log_event(
            db_session,
            site_id="test_site",
            queue="data_entry",
            action_type="advance_queue",
            event_type=AMEEventType.OUTCOME.value,
            proposal_id="P-test-01",
            outcome_success=True,
            observed_time_saved_sec=25.0,
        )
        assert ev.outcome_success is True

    def test_log_event_validates_confidence(self, db_session):
        with pytest.raises(ValueError, match="predicted_confidence"):
            log_event(
                db_session,
                site_id="s",
                queue="q",
                action_type="a",
                event_type="proposal_created",
                predicted_confidence=1.5,
            )

    def test_log_event_validates_safety(self, db_session):
        with pytest.raises(ValueError, match="predicted_safety"):
            log_event(
                db_session,
                site_id="s",
                queue="q",
                action_type="a",
                event_type="proposal_created",
                predicted_safety=-0.1,
            )

    def test_log_event_creates_scope(self, db_session):
        """First event for a scope should auto-create the scope."""
        scope_before = get_scope(db_session, "default", "new_site", "q", "a")
        assert scope_before is None

        log_event(
            db_session,
            site_id="new_site",
            queue="q",
            action_type="a",
            event_type=AMEEventType.PROPOSAL_CREATED.value,
        )

        scope_after = get_scope(db_session, "default", "new_site", "q", "a")
        assert scope_after is not None
        assert scope_after.stage == "observe"


class TestTrustComputation:

    def _log_cycle(self, db, site="trust_test", queue="q", approve=True, success=True):
        """Log a complete proposal -> decision -> outcome cycle."""
        pid = f"P-{id(db)}-{queue}"
        log_event(db, site_id=site, queue=queue, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_CREATED.value,
                  proposal_id=pid, predicted_confidence=0.85,
                  predicted_safety=0.90, predicted_time_saved_sec=30.0)
        log_event(db, site_id=site, queue=queue, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_DECIDED.value,
                  proposal_id=pid,
                  decision="approve" if approve else "reject",
                  decision_by="test_user")
        if approve:
            log_event(db, site_id=site, queue=queue, action_type="advance_queue",
                      event_type=AMEEventType.OUTCOME.value,
                      proposal_id=pid,
                      outcome_success=success,
                      predicted_safety=0.90,
                      observed_time_saved_sec=30.0 if success else 0.0,
                      predicted_time_saved_sec=30.0)

    def test_trust_increases_with_approvals(self, db_session):
        site = "trust_inc"
        for _ in range(6):
            self._log_cycle(db_session, site=site, approve=True, success=True)

        scope = get_scope(db_session, "default", site, "q", "advance_queue")
        assert scope is not None
        assert float(scope.trust_score) > 0.0
        assert float(scope.alignment) > 0.0

    def test_rejections_lower_alignment(self, db_session):
        site = "trust_rej"
        for _ in range(6):
            self._log_cycle(db_session, site=site, approve=False)

        scope = get_scope(db_session, "default", site, "q", "advance_queue")
        assert scope is not None
        assert float(scope.alignment) == pytest.approx(0.0, abs=0.01)

    def test_compute_and_update_scope(self, db_session):
        site = "compute_test"
        for _ in range(5):
            self._log_cycle(db_session, site=site, approve=True, success=True)

        cfg = AMEConfig()
        scope = compute_and_update_scope(
            db_session, "default", site, "q", "advance_queue", None, cfg
        )
        assert float(scope.trust_score) > 0.0
        assert scope.total_proposals >= 5


class TestExecution:

    def test_create_guarded_execution(self, db_session):
        ex = create_execution(
            db_session,
            site_id="exec_test",
            queue="q",
            action_type="advance_queue",
            before_state={"queue": "data_entry"},
            after_state={"queue": "pre_verification"},
            guarded=True,
        )
        db_session.commit()
        assert ex.id is not None
        assert ex.reversible_until is not None
        assert ex.rolled_back is not True

    def test_create_non_guarded_execution(self, db_session):
        ex = create_execution(
            db_session,
            site_id="exec_test",
            queue="q",
            action_type="advance_queue",
            guarded=False,
        )
        db_session.commit()
        assert ex.reversible_until is None

    def test_rollback_execution(self, db_session):
        ex = create_execution(
            db_session,
            site_id="rb_test",
            queue="q",
            action_type="advance_queue",
            guarded=True,
        )
        db_session.commit()

        rolled = rollback_execution(
            db_session, ex.id, reason="test rollback", rolled_back_by="admin"
        )
        assert rolled.rolled_back is True
        assert rolled.rollback_reason == "test rollback"

    def test_rollback_already_rolled_back(self, db_session):
        ex = create_execution(
            db_session,
            site_id="rb2",
            queue="q",
            action_type="advance_queue",
            guarded=True,
        )
        db_session.commit()
        rollback_execution(db_session, ex.id)

        with pytest.raises(ValueError, match="already rolled back"):
            rollback_execution(db_session, ex.id)

    def test_rollback_nonexistent(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            rollback_execution(db_session, 99999)


class TestResolveExecutionMode:

    def test_no_scope_returns_observe(self, db_session):
        mode, meta = resolve_execution_mode(
            db_session, site_id="none", queue="q", action_type="a"
        )
        assert mode == "observe"
        assert meta["reason"] == "no_scope"

    def test_anomaly_downgrades_to_propose(self, db_session):
        # Create scope at guarded_auto level
        scope = AMETrustScope(
            tenant_id="default",
            site_id="anom_test",
            queue="q",
            action_type="a",
            stage=AMEStage.GUARDED_AUTO.value,
            trust_score=0.8,
        )
        db_session.add(scope)
        db_session.commit()

        mode, meta = resolve_execution_mode(
            db_session,
            site_id="anom_test",
            queue="q",
            action_type="a",
            model_confidence=0.3,  # below anomaly threshold
            model_safety=0.9,
        )
        assert mode == "propose"
        assert meta["reason"] == "anomaly_detected"

    def test_normal_returns_current_stage(self, db_session):
        scope = AMETrustScope(
            tenant_id="default",
            site_id="norm_test",
            queue="q",
            action_type="a",
            stage=AMEStage.PROPOSE.value,
            trust_score=0.5,
        )
        db_session.add(scope)
        db_session.commit()

        mode, meta = resolve_execution_mode(
            db_session, site_id="norm_test", queue="q", action_type="a"
        )
        assert mode == "propose"
        assert meta["reason"] == "normal"
