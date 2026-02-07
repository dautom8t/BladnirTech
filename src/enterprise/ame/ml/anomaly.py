"""
AME Anomaly Detector

Predicts: "Is this event sequence normal or unusual?"
Algorithm: IsolationForest (unsupervised)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models import AMEEvent, AMEEventType
from .store import ModelStore
from .features import ANOMALY_FEATURE_NAMES, build_anomaly_window, features_to_array

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Minimum requirements (low for demo; raise for production)
MIN_EVENTS = 20
MIN_DAYS = 0

NORMAL = {
    "is_anomaly": False,
    "anomaly_score": 0.0,
    "anomaly_type": None,
    "contributing_features": [],
    "model_version": 0,
}


class AnomalyDetector:
    """Detects anomalous event patterns using IsolationForest."""

    def __init__(self):
        self.model = None
        self.version: int = 0
        self.metadata: Dict[str, Any] = {}
        self._feature_names: List[str] = list(ANOMALY_FEATURE_NAMES)
        self._baselines: Dict[str, float] = {}
        self._store = ModelStore()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and HAS_SKLEARN

    def load(self, scope_key: str = "global") -> bool:
        """Load trained model from disk."""
        model, meta = self._store.load(scope_key, "anomaly")
        if model is not None:
            self.model = model
            self.metadata = meta
            self.version = meta.get("version", 1)
            self._feature_names = meta.get("feature_names", list(ANOMALY_FEATURE_NAMES))
            self._baselines = meta.get("baselines", {})
            return True
        return False

    def train(
        self,
        windows: List[Dict[str, float]],
        scope_key: str = "global",
    ) -> Dict[str, Any]:
        """Train anomaly detector on windowed event features."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed", "trained": False}

        if len(windows) < 10:
            return {"error": f"Need at least 10 windows, have {len(windows)}", "trained": False}

        X = features_to_array(windows, self._feature_names)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        model.fit(X)

        # Compute baselines (mean of each feature across all windows)
        baselines = {}
        for i, name in enumerate(self._feature_names):
            baselines[name] = float(np.mean(X[:, i]))

        scores = model.decision_function(X)
        labels = model.predict(X)
        anomaly_pct = float(np.mean(labels == -1))

        metrics = {
            "training_windows": len(windows),
            "anomaly_percentage": round(anomaly_pct * 100, 1),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

        self.model = model
        self._baselines = baselines
        save_meta = {
            "training_windows": len(windows),
            "metrics": metrics,
            "feature_names": self._feature_names,
            "baselines": baselines,
        }
        self.version = self._store.save(model, scope_key, "anomaly", save_meta)
        self.metadata = save_meta
        metrics["trained"] = True
        metrics["promoted"] = True
        return metrics

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Check if a single window of features is anomalous."""
        if not self.is_ready:
            return dict(NORMAL)

        try:
            X = np.array([[features.get(k, 0.0) for k in self._feature_names]])

            score = float(self.model.decision_function(X)[0])
            label = int(self.model.predict(X)[0])
            is_anomaly = label == -1

            # Identify contributing features (deviation from baseline)
            contributors = []
            if is_anomaly:
                for i, name in enumerate(self._feature_names):
                    baseline = self._baselines.get(name, 0.0)
                    current = X[0][i]
                    if baseline > 0 and abs(current - baseline) > baseline * 0.5:
                        contributors.append((name, round(current, 3)))

                contributors.sort(key=lambda x: abs(x[1] - self._baselines.get(x[0], 0)), reverse=True)
                contributors = contributors[:3]

            # Determine anomaly type from top contributor
            anomaly_type = None
            if contributors:
                top_feature = contributors[0][0]
                type_map = {
                    "rejection_rate": "rejection_spike",
                    "override_rate": "override_spike",
                    "success_rate": "failure_spike",
                    "event_velocity": "velocity_anomaly",
                    "avg_confidence": "confidence_drop",
                    "avg_safety": "safety_drop",
                }
                anomaly_type = type_map.get(top_feature, "general_anomaly")

            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": round(score, 4),
                "anomaly_type": anomaly_type,
                "contributing_features": contributors,
                "model_version": self.version,
            }
        except Exception as e:
            logger.error(f"Anomaly predict failed: {e}", exc_info=True)
            return dict(NORMAL)

    def check_current_window(
        self,
        db: Session,
        *,
        tenant_id: str = "default",
        site_id: str = "dashboard_demo",
        window_hours: int = 1,
    ) -> Dict[str, Any]:
        """Check the most recent time window for anomalies."""
        if not self.is_ready:
            return dict(NORMAL)

        try:
            now = datetime.utcnow()
            window_start = now - timedelta(hours=window_hours)

            events = (
                db.query(AMEEvent)
                .filter(
                    AMEEvent.tenant_id == tenant_id,
                    AMEEvent.site_id == site_id,
                    AMEEvent.ts >= window_start,
                    AMEEvent.deleted_at.is_(None),
                )
                .order_by(AMEEvent.ts)
                .all()
            )

            if not events:
                return dict(NORMAL)

            features = build_anomaly_window(events, window_start, now)
            if not features:
                return dict(NORMAL)

            return self.predict(features)
        except Exception as e:
            logger.error(f"Anomaly check_current_window failed: {e}", exc_info=True)
            return dict(NORMAL)
