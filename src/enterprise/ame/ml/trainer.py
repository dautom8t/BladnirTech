"""
AME ML Training Orchestration

Coordinates training of all ML models, checks retrain triggers,
and provides CLI interface.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from .decision import DecisionPredictor
from .outcome import OutcomePredictor
from .anomaly import AnomalyDetector
from .store import ModelStore
from .features import (
    get_training_data_decision,
    get_training_data_outcome,
    get_training_data_anomaly,
)

logger = logging.getLogger(__name__)


class AMETrainer:
    """Orchestrates training of all AME ML models."""

    def __init__(
        self,
        tenant_id: str = "default",
        site_id: str = "dashboard_demo",
    ):
        self.tenant_id = tenant_id
        self.site_id = site_id
        self.scope_key = f"{site_id}"
        self._store = ModelStore()

    def train_all(self, db: Session) -> Dict[str, Any]:
        """Train all three models. Returns summary of results."""
        results = {}

        logger.info("Starting AME ML training cycle")

        results["decision"] = self.train_decision(db)
        results["outcome"] = self.train_outcome(db)
        results["anomaly"] = self.train_anomaly(db)
        results["trained_at"] = datetime.utcnow().isoformat() + "Z"

        logger.info(f"Training cycle complete: {results}")
        return results

    def train_decision(self, db: Session) -> Dict[str, Any]:
        """Train decision predictor."""
        logger.info("Training Decision Predictor...")
        features, labels = get_training_data_decision(
            db, tenant_id=self.tenant_id, site_id=self.site_id,
        )

        if not features:
            return {"trained": False, "reason": "no training data"}

        predictor = DecisionPredictor()
        predictor.load(self.scope_key)
        result = predictor.train(features, labels, scope_key=self.scope_key)
        logger.info(f"Decision Predictor: {result}")
        return result

    def train_outcome(self, db: Session) -> Dict[str, Any]:
        """Train outcome predictor."""
        logger.info("Training Outcome Predictor...")
        features, labels = get_training_data_outcome(
            db, tenant_id=self.tenant_id, site_id=self.site_id,
        )

        if not features:
            return {"trained": False, "reason": "no training data"}

        predictor = OutcomePredictor()
        predictor.load(self.scope_key)
        result = predictor.train(features, labels, scope_key=self.scope_key)
        logger.info(f"Outcome Predictor: {result}")
        return result

    def train_anomaly(self, db: Session) -> Dict[str, Any]:
        """Train anomaly detector."""
        logger.info("Training Anomaly Detector...")
        windows = get_training_data_anomaly(
            db, tenant_id=self.tenant_id, site_id=self.site_id,
        )

        if not windows:
            return {"trained": False, "reason": "no training data"}

        detector = AnomalyDetector()
        detector.load("global")
        result = detector.train(windows, scope_key="global")
        logger.info(f"Anomaly Detector: {result}")
        return result

    def check_retrain_needed(self, db: Session) -> Dict[str, Any]:
        """
        Check if any model needs retraining.

        Triggers:
        - 50+ new events since last train
        - 24+ hours since last train
        - Stage change detected
        """
        from ..models import AMEEvent, AMETrustScope

        result = {"needs_retrain": False, "reasons": []}

        # Check each model's metadata
        for model_type in ("decision", "outcome"):
            meta = self._store.get_metadata(self.scope_key, model_type)
            if not meta:
                result["needs_retrain"] = True
                result["reasons"].append(f"{model_type}: never trained")
                continue

            saved_at = meta.get("saved_at", "")
            training_events = meta.get("training_events", 0)

            # Time trigger: 24+ hours since last train
            if saved_at:
                try:
                    last_trained = datetime.fromisoformat(saved_at.rstrip("Z"))
                    hours_ago = (datetime.utcnow() - last_trained).total_seconds() / 3600
                    if hours_ago >= 24:
                        result["needs_retrain"] = True
                        result["reasons"].append(f"{model_type}: {hours_ago:.0f}h since last train")
                except (ValueError, TypeError):
                    pass

            # Event count trigger: 50+ new events
            current_count = (
                db.query(AMEEvent)
                .filter(
                    AMEEvent.tenant_id == self.tenant_id,
                    AMEEvent.site_id == self.site_id,
                    AMEEvent.deleted_at.is_(None),
                )
                .count()
            )
            if current_count - training_events >= 50:
                result["needs_retrain"] = True
                result["reasons"].append(
                    f"{model_type}: {current_count - training_events} new events"
                )

        # Anomaly model
        anomaly_meta = self._store.get_metadata("global", "anomaly")
        if not anomaly_meta:
            result["needs_retrain"] = True
            result["reasons"].append("anomaly: never trained")

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current model status for all models."""
        status = {}
        for model_type in ("decision", "outcome"):
            meta = self._store.get_metadata(self.scope_key, model_type)
            status[model_type] = {
                "trained": bool(meta),
                "version": meta.get("version", 0),
                "saved_at": meta.get("saved_at"),
                "training_events": meta.get("training_events", 0),
                "metrics": meta.get("metrics", {}),
                "feature_importances": meta.get("feature_importances", {}),
            }

        anomaly_meta = self._store.get_metadata("global", "anomaly")
        status["anomaly"] = {
            "trained": bool(anomaly_meta),
            "version": anomaly_meta.get("version", 0),
            "saved_at": anomaly_meta.get("saved_at"),
            "training_windows": anomaly_meta.get("training_windows", 0),
            "metrics": anomaly_meta.get("metrics", {}),
        }

        return status


# =====================================================================
# CLI entry point
# =====================================================================

def _cli_retrain():
    """CLI entry point: python -m src.enterprise.ame.ml.trainer --retrain"""
    import sys
    logging.basicConfig(level=logging.INFO)

    from models.database import get_db_session

    try:
        db = next(get_db_session())
    except Exception:
        # Fallback: create session directly
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./BladnirTech.db")
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

    trainer = AMETrainer()

    if "--status" in sys.argv:
        print(json.dumps(trainer.get_status(), indent=2))
    elif "--check" in sys.argv:
        print(json.dumps(trainer.check_retrain_needed(db), indent=2))
    else:
        results = trainer.train_all(db)
        print(json.dumps(results, indent=2, default=str))

    db.close()


if __name__ == "__main__":
    _cli_retrain()
