# AME Machine Learning Spec — Adaptive Model Evolution

## Status: SPEC (not implemented)

This document specifies the ML layer that sits on top of the existing AME event
data spine. The goal: make the system actually **learn** from every human
decision, outcome, and override — then feed predictions back into runtime
behavior.

---

## What Exists Today (statistical thresholding)

```
Events → Weighted metrics → Hardcoded thresholds → Stage → Behavior
```

- Trust score is a fixed-weight composite (35/25/25/15)
- Stage thresholds are constants (0.40, 0.60, 0.75, 0.85)
- Anomaly detection is two static checks (confidence < 0.50, safety < 0.60)
- No model is trained, no predictions improve over time

## What This Adds (actual ML)

```
Events → Feature extraction → Trained models → Predictions → Runtime behavior
                                    ↑                              ↓
                              Retrain loop ←──── Outcome feedback ──┘
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   AME ML Layer                        │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Decision    │  │  Outcome     │  │  Anomaly    │ │
│  │  Predictor   │  │  Predictor   │  │  Detector   │ │
│  │  (approve?)  │  │  (succeed?)  │  │  (normal?)  │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                │                  │         │
│  ┌──────┴────────────────┴──────────────────┴──────┐ │
│  │            Feature Engineering                   │ │
│  │  AMEEvent × AMETrustScope × context_json         │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                             │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │         Model Store (joblib on disk)             │ │
│  │  data/models/{scope_key}/decision_v{n}.pkl       │ │
│  │  data/models/{scope_key}/outcome_v{n}.pkl        │ │
│  │  data/models/global/anomaly_v{n}.pkl             │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │         Retrain Scheduler                        │ │
│  │  Triggers: event count, time elapsed, drift      │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
         │                    ▲
         ▼                    │
┌──────────────────┐  ┌──────────────────┐
│  AME Service     │  │  AME Events DB   │
│  (existing)      │  │  (existing)      │
│  resolve_mode()  │  │  ame_events      │
│  log_event()     │  │  ame_trust_scopes│
└──────────────────┘  └──────────────────┘
```

---

## Model 1: Decision Predictor

**Question:** "Will a human approve or reject this proposal?"

### Why It Matters
Today, every proposal requires human review until GUARDED_AUTO. A decision
predictor lets the system pre-score proposals so humans can focus on the ones
the model is uncertain about, and auto-approve the ones it's confident about.

### Training Data
- **Source:** AMEEvent rows where `event_type = 'proposal_decided'`
- **Label:** `decision` (binary: approve=1, reject=0, defer excluded)
- **Features:**

| Feature | Source | Type |
|---------|--------|------|
| queue | AMEEvent.queue | categorical |
| action_type | AMEEvent.action_type | categorical |
| predicted_confidence | AMEEvent.predicted_confidence | float |
| predicted_safety | AMEEvent.predicted_safety | float |
| predicted_time_saved | AMEEvent.predicted_time_saved_sec | int |
| scope_trust | AMETrustScope.trust_score | float |
| scope_stage | AMETrustScope.stage | categorical |
| scope_reliability | AMETrustScope.reliability | float |
| scope_alignment | AMETrustScope.alignment | float |
| approval_rate_7d | computed: approvals/total in last 7 days | float |
| rejection_streak | computed: consecutive recent rejections | int |
| hour_of_day | AMEEvent.ts.hour | int |
| events_last_hour | computed: event count in last 60 min | int |
| insurance_result | context_json.insurance_result | categorical |
| scenario_type | context_json.scenario_id | categorical |

### Model
- **Algorithm:** `sklearn.ensemble.GradientBoostingClassifier`
- **Why:** Handles mixed feature types, works well with small datasets (< 10K rows),
  interpretable via feature importances
- **Fallback:** `sklearn.linear_model.SGDClassifier` for online/incremental learning
  when event volume is high

### Output
```python
{
    "approve_probability": 0.92,    # P(human approves)
    "confidence_band": "high",      # high (>0.85), medium (0.5-0.85), low (<0.5)
    "top_factors": [                 # SHAP-like explanation
        ("predicted_safety", +0.15),
        ("scope_reliability", +0.12),
        ("insurance_result=rejected", -0.08),
    ]
}
```

### Integration Point
- `propose_automation()` in `bladnir_dashboard.py` calls the model before creating
  the proposal
- Response includes `ml_approve_probability` so the UI can show:
  - Green badge: "AI predicts approval (92%)"
  - Yellow badge: "Uncertain — needs review"
  - Red badge: "AI predicts rejection (78%)"
- At CONDITIONAL_AUTO stage: if `approve_probability > 0.90`, auto-approve
  without human review

### Minimum Training Data
- 50 decided proposals (approve or reject)
- At least 10 rejections (otherwise the model has no negative examples)

---

## Model 2: Outcome Predictor

**Question:** "If we execute this action, will it succeed?"

### Why It Matters
Today, `predicted_confidence` and `predicted_safety` are static values passed
in by the caller (hardcoded 0.86 and 0.92 in auto-step). A trained outcome
predictor replaces these with real predictions based on historical outcomes.

### Training Data
- **Source:** AMEEvent rows where `event_type = 'outcome'`
- **Label:** `outcome_success` (binary: True=1, False=0)
- **Features:**

| Feature | Source | Type |
|---------|--------|------|
| queue | AMEEvent.queue | categorical |
| action_type | AMEEvent.action_type | categorical |
| from_queue | context_json.transition (parsed) | categorical |
| to_queue | context_json.transition (parsed) | categorical |
| scope_trust | AMETrustScope.trust_score | float |
| scope_override_rate | AMETrustScope.override_rate | float |
| recent_success_rate | computed: successes/total in last 20 outcomes | float |
| recent_rollback_count | computed: rollbacks in last 7 days | int |
| governance_gate_enabled | governance gate status | boolean |
| execution_count_today | computed: executions in last 24h | int |
| insurance_result | context_json | categorical |

### Model
- **Algorithm:** `sklearn.ensemble.GradientBoostingClassifier`
- **Secondary:** Calibrated with `CalibratedClassifierCV` so probabilities are
  reliable (important: we use the probability directly as `predicted_confidence`)

### Output
```python
{
    "success_probability": 0.88,
    "safety_score": 0.91,          # separate safety model or derived
    "predicted_time_saved_sec": 35, # regression side-model
    "risk_factors": [
        ("recent_rollback_count=2", -0.12),
        ("insurance_result=rejected", -0.08),
    ]
}
```

### Integration Point
- Replaces the hardcoded `predicted_confidence=0.86, predicted_safety=0.92`
  in `auto_step` and `propose_automation`
- Feeds into `resolve_execution_mode()` for anomaly detection
- If `success_probability < anomaly_min_confidence`, the system automatically
  falls back to PROPOSE (already implemented in resolve_execution_mode)

### Minimum Training Data
- 30 outcomes with at least 5 failures
- Until threshold is met, fall back to the existing static values

---

## Model 3: Anomaly Detector

**Question:** "Is this event sequence normal or unusual?"

### Why It Matters
Today, anomaly detection is two static checks. A learned anomaly detector can
catch patterns like: "3 rejections in a row on this queue" or "execution time
suddenly doubled" or "override rate spiking."

### Training Data
- **Source:** All AMEEvent rows, windowed into sequences per scope
- **Approach:** Unsupervised — no labels needed
- **Features (per 1-hour window):**

| Feature | Computation |
|---------|-------------|
| proposal_count | count of proposal_created events |
| rejection_rate | rejections / decisions |
| override_rate | overrides / outcomes |
| avg_confidence | mean predicted_confidence |
| avg_safety | mean predicted_safety |
| success_rate | successes / outcomes |
| avg_execution_time | mean execution_duration_ms |
| unique_actors | distinct decision_by values |
| event_velocity | events per minute |

### Model
- **Algorithm:** `sklearn.ensemble.IsolationForest`
- **Why:** Works well with small datasets, no labels needed, fast inference
- **Alternative:** `sklearn.covariance.EllipticEnvelope` for Gaussian-assumption
  anomaly detection

### Output
```python
{
    "is_anomaly": True,
    "anomaly_score": -0.42,         # negative = more anomalous
    "anomaly_type": "rejection_spike",
    "contributing_features": [
        ("rejection_rate", 0.80),    # normally ~0.10
        ("override_rate", 0.30),     # normally ~0.02
    ]
}
```

### Integration Point
- Called inside `resolve_execution_mode()` alongside the existing static checks
- If anomaly detected AND stage is GUARDED_AUTO+, downgrade to PROPOSE
- Log as `AMEEventType.ANOMALY` with anomaly details in `metadata_json`
- AME Trust Dashboard shows anomaly alerts

### Minimum Training Data
- 100 events across at least 3 days (needs baseline of "normal")
- Until threshold is met, fall back to existing static anomaly checks

---

## Model Store & Versioning

```
data/
  models/
    global/
      anomaly_v1.pkl
      anomaly_v2.pkl          # kept for rollback
      anomaly_metadata.json   # {version, trained_at, event_count, metrics}
    dashboard_demo/
      contact_manager/
        decision_v1.pkl
        outcome_v1.pkl
        metadata.json
      data_entry/
        decision_v1.pkl
        outcome_v1.pkl
        metadata.json
```

### Metadata per model:
```json
{
    "version": 3,
    "trained_at": "2026-02-07T14:30:00Z",
    "training_events": 247,
    "training_window": "2026-01-01 to 2026-02-07",
    "metrics": {
        "accuracy": 0.89,
        "precision": 0.91,
        "recall": 0.87,
        "f1": 0.89,
        "auc_roc": 0.94
    },
    "feature_importances": {
        "predicted_confidence": 0.22,
        "scope_trust": 0.18,
        "queue": 0.15
    },
    "min_events_for_prediction": 50
}
```

---

## Retrain Strategy

### Triggers (any one fires a retrain):

| Trigger | Condition | Rationale |
|---------|-----------|-----------|
| Event count | 50 new events since last train | Enough new data to matter |
| Time elapsed | 24 hours since last train | Catch temporal drift |
| Performance drift | Live accuracy drops 10% below training accuracy | Model degrading |
| Stage change | Any scope changes stage | Behavior changed, revalidate |

### Retrain Process:
1. Query last N events (configurable, default 500)
2. Extract features
3. Train new model version
4. Evaluate on holdout (last 20% of events)
5. If new model beats old model by > 1% accuracy, promote to active
6. If new model is worse, keep old model, log warning
7. Save model artifact + metadata

### Cold Start (< minimum training data):
- Decision predictor: return `{"approve_probability": 0.5, "confidence_band": "cold_start"}`
- Outcome predictor: return hardcoded values (0.86 confidence, 0.92 safety)
- Anomaly detector: use existing static threshold checks
- System operates identically to today until enough data accumulates

---

## Implementation Plan

### Phase 1: Feature Engineering + Training Pipeline (foundation)
**Files:** `src/enterprise/ame/ml/features.py`, `src/enterprise/ame/ml/trainer.py`

- Feature extraction from AMEEvent + AMETrustScope
- Train/evaluate pipeline for all 3 models
- Model persistence to disk (joblib)
- CLI command: `python -m src.enterprise.ame.ml.trainer --retrain`

### Phase 2: Decision Predictor (highest impact)
**Files:** `src/enterprise/ame/ml/decision.py`

- Integrate into `propose_automation()`: score proposals before creation
- Return `ml_approve_probability` in API response
- Dashboard UI: show approval prediction badge on suggested actions
- At CONDITIONAL_AUTO: auto-approve if probability > 0.90

### Phase 3: Outcome Predictor (replaces hardcoded values)
**Files:** `src/enterprise/ame/ml/outcome.py`

- Replace hardcoded `predicted_confidence=0.86` in auto-step
- Feed real predictions into `resolve_execution_mode()` anomaly detection
- Log predicted vs actual for continuous calibration

### Phase 4: Anomaly Detector (safety net)
**Files:** `src/enterprise/ame/ml/anomaly.py`

- Train IsolationForest on windowed event features
- Integrate into `resolve_execution_mode()` alongside static checks
- Log anomalies as AME events
- Dashboard alert panel for active anomalies

### Phase 5: Retrain Loop (production readiness)
**Files:** `src/enterprise/ame/ml/scheduler.py`

- Background task that checks retrain triggers
- Automatic model versioning and promotion
- Drift monitoring (compare live predictions vs outcomes)
- API endpoint: `GET /ame/ml/status` — model versions, training metrics, drift

---

## Dependencies

```
# Add to requirements.txt
scikit-learn>=1.4
joblib>=1.3          # model persistence (included with scikit-learn)
numpy>=1.26          # scikit-learn dependency
```

No GPU required. No TensorFlow/PyTorch. scikit-learn is lightweight (~30MB)
and runs inference in microseconds.

---

## Patent Alignment

| Patent Claim | ML Coverage |
|---|---|
| ML decision engines | Decision Predictor (approve/reject) + Outcome Predictor (succeed/fail) |
| Learns from human oversight | Decision model trains on approve/reject history |
| Adapts over time | Retrain loop + exponential decay weighting |
| Safety calibration | Outcome predictor replaces static safety scores |
| Anomaly detection | IsolationForest on event sequences |
| Reduces human workload | CONDITIONAL_AUTO auto-approves high-confidence proposals |
| Audit trail for decisions | ML predictions logged in AMEEvent.metadata_json |

---

## Example Flow (after ML is live)

```
1. User clicks "Auto-step" on a case in data_entry queue

2. System calls Outcome Predictor:
   → success_probability: 0.91, safety: 0.88
   (trained on 200 historical outcomes for this queue)

3. System calls resolve_execution_mode():
   → stage: conditional_auto, no anomalies detected

4. System calls Decision Predictor:
   → approve_probability: 0.94, confidence: high
   → top_factors: high scope_trust (0.82), no recent rejections

5. Since stage=conditional_auto AND approve_probability > 0.90:
   → Auto-approve, skip human review
   → Create execution record with 15-min rollback window

6. Action executes, outcome logged:
   → outcome_success: True, observed_time_saved: 28 sec

7. Next retrain cycle:
   → This outcome becomes training data for both models
   → Trust score updates, stage may progress
```
