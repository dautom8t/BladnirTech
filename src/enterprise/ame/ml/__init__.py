"""
AME Machine Learning Layer

Provides trained ML models for:
- Decision prediction (will a human approve?)
- Outcome prediction (will this action succeed?)
- Anomaly detection (is this event sequence normal?)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .decision import DecisionPredictor
from .outcome import OutcomePredictor
from .anomaly import AnomalyDetector
from .store import ModelStore

logger = logging.getLogger(__name__)

# Module-level model cache (keyed by scope)
_decision_cache: Dict[str, DecisionPredictor] = {}
_outcome_cache: Dict[str, OutcomePredictor] = {}
_anomaly_instance: Optional[AnomalyDetector] = None


def get_decision_predictor(scope_key: str = "global") -> DecisionPredictor:
    """Get or load Decision Predictor for a scope."""
    if scope_key not in _decision_cache:
        p = DecisionPredictor()
        p.load(scope_key)
        _decision_cache[scope_key] = p
    return _decision_cache[scope_key]


def get_outcome_predictor(scope_key: str = "global") -> OutcomePredictor:
    """Get or load Outcome Predictor for a scope."""
    if scope_key not in _outcome_cache:
        p = OutcomePredictor()
        p.load(scope_key)
        _outcome_cache[scope_key] = p
    return _outcome_cache[scope_key]


def get_anomaly_detector() -> AnomalyDetector:
    """Get or load global Anomaly Detector."""
    global _anomaly_instance
    if _anomaly_instance is None:
        _anomaly_instance = AnomalyDetector()
        _anomaly_instance.load("global")
    return _anomaly_instance


def clear_cache():
    """Clear all cached models (used after retrain or reset)."""
    global _anomaly_instance
    _decision_cache.clear()
    _outcome_cache.clear()
    _anomaly_instance = None


__all__ = [
    "DecisionPredictor",
    "OutcomePredictor",
    "AnomalyDetector",
    "ModelStore",
    "get_decision_predictor",
    "get_outcome_predictor",
    "get_anomaly_detector",
    "clear_cache",
]
