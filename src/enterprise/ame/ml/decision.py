"""
AME Decision Predictor

Predicts: "Will a human approve or reject this proposal?"
Algorithm: GradientBoostingClassifier (scikit-learn)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .store import ModelStore
from .features import (
    DECISION_FEATURE_NAMES,
    build_decision_features,
    compute_decision_context,
    features_to_array,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Minimum requirements (low for demo; raise for production)
MIN_DECIDED = 15
MIN_REJECTIONS = 3

COLD_START = {
    "approve_probability": 0.5,
    "confidence_band": "cold_start",
    "top_factors": [],
    "model_version": 0,
}


class DecisionPredictor:
    """Predicts whether a human will approve or reject a proposal."""

    def __init__(self):
        self.model = None
        self.version: int = 0
        self.metadata: Dict[str, Any] = {}
        self._feature_names: List[str] = list(DECISION_FEATURE_NAMES)
        self._store = ModelStore()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and HAS_SKLEARN

    def load(self, scope_key: str) -> bool:
        """Load trained model from disk."""
        model, meta = self._store.load(scope_key, "decision")
        if model is not None:
            self.model = model
            self.metadata = meta
            self.version = meta.get("version", 1)
            self._feature_names = meta.get("feature_names", list(DECISION_FEATURE_NAMES))
            return True
        return False

    def train(
        self,
        features_list: List[Dict[str, float]],
        labels: List[int],
        scope_key: str = "global",
    ) -> Dict[str, Any]:
        """Train model. Returns metrics dict."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed", "trained": False}

        n = len(features_list)
        rejections = sum(1 for l in labels if l == 0)

        if n < MIN_DECIDED:
            return {"error": f"Need {MIN_DECIDED} decisions, have {n}", "trained": False}
        if rejections < MIN_REJECTIONS:
            return {"error": f"Need {MIN_REJECTIONS} rejections, have {rejections}", "trained": False}

        X = features_to_array(features_list, self._feature_names)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0

        importances = dict(zip(self._feature_names, model.feature_importances_.tolist()))

        # Promote if new model is within 1% of old
        old_acc = self.metadata.get("metrics", {}).get("accuracy", 0)
        if metrics["accuracy"] >= old_acc - 0.01:
            self.model = model
            save_meta = {
                "training_events": n,
                "metrics": metrics,
                "feature_importances": importances,
                "feature_names": self._feature_names,
                "min_events_for_prediction": MIN_DECIDED,
            }
            self.version = self._store.save(model, scope_key, "decision", save_meta)
            self.metadata = save_meta
            metrics["promoted"] = True
        else:
            metrics["promoted"] = False

        metrics["trained"] = True
        return metrics

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict from a pre-built feature dict."""
        if not self.is_ready:
            return dict(COLD_START)

        try:
            X = np.array([[features.get(k, 0.0) for k in self._feature_names]])
            proba = self.model.predict_proba(X)[0]
            approve_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

            if approve_prob > 0.85:
                band = "high"
            elif approve_prob > 0.50:
                band = "medium"
            else:
                band = "low"

            # Top contributing features
            imp = self.model.feature_importances_
            top_idx = np.argsort(imp)[::-1][:3]
            factors = []
            for idx in top_idx:
                name = self._feature_names[idx] if idx < len(self._feature_names) else f"f{idx}"
                val = X[0][idx]
                direction = +1 if val > 0 else -1
                factors.append((name, round(float(imp[idx]) * direction, 4)))

            return {
                "approve_probability": round(approve_prob, 4),
                "confidence_band": band,
                "top_factors": factors,
                "model_version": self.version,
            }
        except Exception as e:
            logger.error(f"Decision predict failed: {e}", exc_info=True)
            return dict(COLD_START)

    def predict_for_proposal(
        self,
        db,
        *,
        queue: str,
        action_type: str,
        confidence: float,
        safety: float,
        time_saved: float = 0.0,
        insurance_result: str = "unknown",
        tenant_id: str = "default",
        site_id: str = "dashboard_demo",
    ) -> Dict[str, Any]:
        """High-level: predict approval for a proposal (queries DB for context)."""
        if not self.is_ready:
            return dict(COLD_START)

        try:
            ctx = compute_decision_context(db, tenant_id, site_id, queue, action_type)
            features = build_decision_features(
                queue=queue,
                action_type=action_type,
                predicted_confidence=confidence,
                predicted_safety=safety,
                predicted_time_saved=time_saved,
                insurance_result=insurance_result,
                **ctx,
            )
            return self.predict(features)
        except Exception as e:
            logger.error(f"Decision predict_for_proposal failed: {e}", exc_info=True)
            return dict(COLD_START)
