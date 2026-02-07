"""
AME Outcome Predictor

Predicts: "If we execute this action, will it succeed?"
Algorithm: GradientBoostingClassifier with CalibratedClassifierCV
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .store import ModelStore
from .features import (
    OUTCOME_FEATURE_NAMES,
    build_outcome_features,
    compute_outcome_context,
    features_to_array,
)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Minimum requirements (low for demo; raise for production)
MIN_OUTCOMES = 10
MIN_FAILURES = 2

# Cold start: identical to existing hardcoded values
COLD_START = {
    "success_probability": 0.86,
    "safety_score": 0.92,
    "risk_factors": [],
    "model_version": 0,
    "cold_start": True,
}


class OutcomePredictor:
    """Predicts whether an action will succeed."""

    def __init__(self):
        self.model = None
        self.version: int = 0
        self.metadata: Dict[str, Any] = {}
        self._feature_names: List[str] = list(OUTCOME_FEATURE_NAMES)
        self._store = ModelStore()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and HAS_SKLEARN

    def load(self, scope_key: str) -> bool:
        """Load trained model from disk."""
        model, meta = self._store.load(scope_key, "outcome")
        if model is not None:
            self.model = model
            self.metadata = meta
            self.version = meta.get("version", 1)
            self._feature_names = meta.get("feature_names", list(OUTCOME_FEATURE_NAMES))
            return True
        return False

    def train(
        self,
        features_list: List[Dict[str, float]],
        labels: List[int],
        scope_key: str = "global",
    ) -> Dict[str, Any]:
        """Train model with probability calibration."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not installed", "trained": False}

        n = len(features_list)
        failures = sum(1 for l in labels if l == 0)

        if n < MIN_OUTCOMES:
            return {"error": f"Need {MIN_OUTCOMES} outcomes, have {n}", "trained": False}
        if failures < MIN_FAILURES:
            return {"error": f"Need {MIN_FAILURES} failures, have {failures}", "trained": False}

        X = features_to_array(features_list, self._feature_names)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Base model
        base = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )

        # Calibrate probabilities — adapt cv to available minority class samples
        min_class_count = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
        cv_folds = max(2, min(3, min_class_count))

        if min_class_count >= 2:
            model = CalibratedClassifierCV(base, cv=cv_folds, method="sigmoid")
            model.fit(X_train, y_train)
        else:
            # Too few minority samples for calibration; use base model directly
            base.fit(X_train, y_train)
            model = base

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

        # Feature importances from base estimator
        try:
            if hasattr(model, 'calibrated_classifiers_'):
                base_model = model.calibrated_classifiers_[0].estimator
            else:
                base_model = model
            importances = dict(zip(self._feature_names, base_model.feature_importances_.tolist()))
        except (AttributeError, IndexError):
            importances = {}

        old_acc = self.metadata.get("metrics", {}).get("accuracy", 0)
        if metrics["accuracy"] >= old_acc - 0.01:
            self.model = model
            save_meta = {
                "training_events": n,
                "metrics": metrics,
                "feature_importances": importances,
                "feature_names": self._feature_names,
                "min_events_for_prediction": MIN_OUTCOMES,
            }
            self.version = self._store.save(model, scope_key, "outcome", save_meta)
            self.metadata = save_meta
            metrics["promoted"] = True
        else:
            metrics["promoted"] = False

        metrics["trained"] = True
        return metrics

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict from pre-built feature dict."""
        if not self.is_ready:
            return dict(COLD_START)

        try:
            X = np.array([[features.get(k, 0.0) for k in self._feature_names]])
            proba = self.model.predict_proba(X)[0]
            success_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

            # Derive safety score from success probability (conservative: slightly below)
            safety = min(success_prob * 0.95 + 0.05, 1.0)

            # Risk factors from feature importances
            risk_factors = []
            try:
                if hasattr(self.model, 'calibrated_classifiers_'):
                    base_model = self.model.calibrated_classifiers_[0].estimator
                else:
                    base_model = self.model
                imp = base_model.feature_importances_
                for idx in np.argsort(imp)[::-1][:3]:
                    name = self._feature_names[idx] if idx < len(self._feature_names) else f"f{idx}"
                    val = X[0][idx]
                    if val < 0.3:  # Flag low-value features as risks
                        risk_factors.append((name, round(-float(imp[idx]), 4)))
            except (AttributeError, IndexError):
                pass

            return {
                "success_probability": round(success_prob, 4),
                "safety_score": round(safety, 4),
                "risk_factors": risk_factors,
                "model_version": self.version,
                "cold_start": False,
            }
        except Exception as e:
            logger.error(f"Outcome predict failed: {e}", exc_info=True)
            return dict(COLD_START)

    def predict_for_action(
        self,
        db,
        *,
        queue: str,
        action_type: str,
        from_queue: str = "",
        to_queue: str = "",
        insurance_result: str = "unknown",
        tenant_id: str = "default",
        site_id: str = "dashboard_demo",
    ) -> Dict[str, Any]:
        """High-level: predict outcome for an action (queries DB for context)."""
        if not self.is_ready:
            return dict(COLD_START)

        try:
            ctx = compute_outcome_context(db, tenant_id, site_id, queue, action_type)
            features = build_outcome_features(
                queue=queue,
                action_type=action_type,
                from_queue=from_queue,
                to_queue=to_queue,
                insurance_result=insurance_result,
                **ctx,
            )
            return self.predict(features)
        except Exception as e:
            logger.error(f"Outcome predict_for_action failed: {e}", exc_info=True)
            return dict(COLD_START)
