


from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
from typing import Dict, Any, List
import os
import uuid
import logging
log = logging.getLogger("uvicorn.error")
_ame_log = logging.getLogger("bladnir.ame")
from enterprise import governance

from models.database import get_db
from services import workflow as workflow_service
from models.schemas import EventCreate
from src.enterprise.ame.service import (
    log_event as _ame_log_event,
    resolve_execution_mode as _ame_resolve,
    create_execution as _ame_create_execution,
)
from src.enterprise.ame.models import AMEEventType
from src.enterprise.ame.ml import (
    get_decision_predictor as _ml_decision,
    get_outcome_predictor as _ml_outcome,
    clear_cache as _ml_clear_cache,
)

router = APIRouter(tags=["dashboard"])

from typing import Dict

# =============================
# Live Activity Feed (narrates actions in plain English)
# =============================

ACTIVITY_FEED: List[Dict[str, Any]] = []
_FEED_MAX = 200  # keep last 200 entries

def _narrate(message: str, category: str = "info", detail: str = "", meta: Dict[str, Any] | None = None):
    """Add a narrated event to the activity feed. Categories: info, proposal, decision, execution, trust, gate, ml, warning."""
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "message": message,
        "category": category,
        "detail": detail,
    }
    if meta:
        entry["meta"] = meta
    ACTIVITY_FEED.append(entry)
    if len(ACTIVITY_FEED) > _FEED_MAX:
        ACTIVITY_FEED.pop(0)

def _stage_label(stage: str) -> str:
    return {
        "observe": "OBSERVE — AI watches, humans do everything",
        "propose": "PROPOSE — AI suggests actions, humans decide",
        "guarded_auto": "GUARDED AUTO — AI acts with rollback window",
        "conditional_auto": "CONDITIONAL AUTO — AI acts when thresholds pass",
        "full_auto": "FULL AUTO — AI acts autonomously",
    }.get(stage, stage or "unknown")

def _short_stage(stage: str) -> str:
    return {"observe": "Observe", "propose": "Propose", "guarded_auto": "Guarded Auto",
            "conditional_auto": "Conditional Auto", "full_auto": "Full Auto"}.get(stage, stage or "—")

# Single source of truth: transitions + the gate key required
TRANSITIONS = {
    "contact_manager":  {"to": "inbound_comms",     "gate": "kroger.prescriber_approval_to_data_entry"},
    "inbound_comms":    {"to": "data_entry",        "gate": "kroger.prescriber_approval_to_data_entry"},
    "data_entry":       {"to": "pre_verification",  "gate": "kroger.data_entry_to_preverify_insurance"},
    "pre_verification": {"to": "dispensing",        "gate": "kroger.preverify_to_access_granted"},
    "dispensing":       {"to": "verification",      "gate": "kroger.preverify_to_access_granted"},
}

def next_queue_for(cur: str):
    """Preview-only (no gating). Used for Suggested Actions."""
    t = TRANSITIONS.get(cur)
    return t["to"] if t else None

def gate_for_transition(from_q: str, to_q: str):
    t = TRANSITIONS.get(from_q)
    if not t or t["to"] != to_q:
        return None
    return t["gate"]

def step_queue(cur: str):
    """Enforced step (requires governance authorization)."""
    t = TRANSITIONS.get(cur)
    if not t:
        return cur, None
    governance.require_authorized(t["gate"])
    return t["to"], t["gate"]

# =============================
# AME Trust Integration
# =============================

_AME_SITE = "dashboard_demo"


def _ame_safe(db, **kwargs):
    """Log AME trust event. Non-blocking: failures never break the dashboard."""
    try:
        return _ame_log_event(db, tenant_id="default", site_id=_AME_SITE, **kwargs)
    except Exception as exc:
        _ame_log.warning("AME event skipped: %s", exc)
        return None


def _ame_trust_info(db, queue, action_type="advance_queue"):
    """Get current AME trust stage for a scope. Returns dict safe for JSON response."""
    try:
        mode, meta = _ame_resolve(
            db, site_id=_AME_SITE, queue=queue, action_type=action_type,
        )
        return {"stage": mode, "trust": meta.get("trust", 0.0), "meta": meta}
    except Exception:
        return {"stage": "observe", "trust": 0.0}


def _ame_resolve_safe(db, queue, action_type="advance_queue", confidence=0.75, safety=0.85):
    """Resolve AME execution mode. Returns (stage_str, meta_dict) — never throws."""
    try:
        return _ame_resolve(
            db, site_id=_AME_SITE, queue=queue, action_type=action_type,
            model_confidence=confidence, model_safety=safety,
        )
    except Exception:
        return "observe", {"trust": 0.0, "stage": "observe", "reason": "error"}


@router.get("/dashboard/api/automation")
def dashboard_get_automation():
    # UI expects {authorizations:{key:bool}}
    gates = governance.list_gates()
    keys = {t["gate"] for t in TRANSITIONS.values()}
    return {"authorizations": {k: bool(gates.get(k, {}).get("enabled", False)) for k in keys}}

@router.post("/dashboard/api/automation")
def dashboard_set_automation(
    transition_key: str = Body(..., embed=True),
    enabled: bool = Body(..., embed=True),
    decided_by: str = Body("human", embed=True),
    note: str = Body("", embed=True),
):
    # Look up human-readable names for the gate
    _gate_names = {
        "kroger.prescriber_approval_to_data_entry": "Prescriber \u2192 Data Entry",
        "kroger.data_entry_to_preverify_insurance": "Data Entry \u2192 Pre-Verification",
        "kroger.preverify_to_access_granted": "Pre-Verification \u2192 Dispensing",
    }
    gate_label = _gate_names.get(transition_key, transition_key)

    if enabled:
        governance.authorize_gate(transition_key, actor=decided_by, note=note)
        _narrate(
            f"Gate AUTHORIZED: {gate_label}",
            "gate",
            f"Human operator enabled the governance gate for this transition. "
            f"Cases can now advance through this step — but only if the AME trust stage allows it or a human approves the proposal.",
        )
    else:
        governance.revoke_gate(transition_key, actor=decided_by, note=note)
        _narrate(
            f"Gate REVOKED: {gate_label}",
            "gate",
            f"Human operator disabled this governance gate. "
            f"No cases can advance through this transition until it is re-authorized, regardless of AME trust stage.",
        )
    return dashboard_get_automation()

DEMO_SCENARIOS = {
    "happy_path": {
        "label": "Happy Path — Standard Refill",
        "description": "A routine atorvastatin refill with valid insurance. Demonstrates the full queue progression from inbound comms through dispensing, with governance gates controlling each transition.",
        "start_queue": "inbound_comms",
        "insurance_result": "accepted",
        "refills_ok": True,
        "has_insurance": True,
        "seed_narrative": "Standard prescription refill received. Insurance is active and accepted. This case will flow through each queue — watch how governance gates and AME trust stages control every transition.",
    },
    "insurance_rejected_outdated": {
        "label": "Insurance Rejected — Patient Outreach",
        "description": "Insurance adjudication fails due to outdated info. The system must pause at pre-verification and trigger patient outreach — showing how governance gates prevent bad transitions and protect patient safety.",
        "start_queue": "data_entry",
        "insurance_result": "rejected",
        "reject_reason": "Outdated or missing insurance information",
        "refills_ok": True,
        "has_insurance": True,
        "patient_message": {
            "type": "insurance_update_request",
            "template": "Your insurance info appears outdated or missing. Please upload/confirm your active plan to continue."
        },
        "seed_narrative": "Insurance rejected — outdated information on file. Case starts at Data Entry where the technician discovers the issue. The governance gate to Pre-Verification will block advancement until resolved.",
    },
    "prior_auth_required": {
        "label": "Prior Authorization — Multi-Day Wait",
        "description": "The payer requires prior authorization before dispensing. This demonstrates how PactGate handles waiting states — the case stays governed while PA is pending, and AME learns from the delay patterns.",
        "start_queue": "pre_verification",
        "insurance_result": "pa_required",
        "refills_ok": True,
        "has_insurance": True,
        "pa": {"eta_days": 2},
        "patient_message": {
            "type": "prior_auth_notice",
            "template": "Your plan requires prior authorization. We've initiated PA; we'll update you as soon as we hear back."
        },
        "seed_narrative": "Prior authorization required by payer (est. 2 business days). Case starts at Pre-Verification where the pharmacist reviews and initiates the PA process. Trust decisions at this queue are critical — PA requests are high-stakes.",
    },
    "no_refills_prescriber": {
        "label": "No Refills — Prescriber Request",
        "description": "Patient has zero refills remaining. The system generates an outbound request to the prescriber. Shows the contact manager queue in action and how AME tracks prescriber response SLAs.",
        "start_queue": "contact_manager",
        "insurance_result": "accepted",
        "refills_ok": False,
        "has_insurance": True,
        "prescriber_request": {
            "type": "refill_request",
            "template": "No refills remaining. Refill request sent to prescriber."
        },
        "seed_narrative": "No refills remaining — contacting prescriber. Case starts at Contact Manager where the outbound request is generated. Watch how this queue's trust stage evolves as the system learns prescriber response patterns.",
    },
    "no_insurance_discount_card": {
        "label": "Cash Pay — Discount Card Applied",
        "description": "Patient has no insurance on file. The system automatically applies a DemoRxSaver discount card. Demonstrates how AME can learn to auto-apply discount programs as trust grows.",
        "start_queue": "data_entry",
        "insurance_result": "no_insurance",
        "refills_ok": True,
        "has_insurance": False,
        "discount_card": {"program": "DemoRxSaver", "bin": "999999", "pcn": "DEMO", "group": "SAVER", "member": "DEMO1234"},
        "patient_message": {
            "type": "discount_card_applied",
            "template": "No active insurance found. We applied a discount card to help complete your prescription."
        },
        "seed_narrative": "No insurance on file — discount card applied. Case starts at Data Entry where the discount program is matched. As AME trust grows, this discount-card application can become fully automated.",
    },
}
DEMO_ROWS = []
DEMO_BY_ID = {}

# =============================
# Authorized Automation (DEMO / in-memory)
# =============================

DEMO_PROPOSALS: List[Dict[str, Any]] = []
DEMO_PROPOSAL_BY_ID: Dict[str, Dict[str, Any]] = {}

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _mk_proposal(case_id: int, action: Dict[str, Any]) -> Dict[str, Any]:
    pid = "P-" + uuid.uuid4().hex[:10]
    now = _now_iso()
    row = {
        "id": pid,
        "case_id": int(case_id),

        # ✅ ACTION
        "action": action,

        # ✅ STATUS
        "status": "pending",

        # ✅ ATTRIBUTION
        "proposed_by": "system",
        "proposed_at": now,
        "created_at": now,  # FIXED: Added missing created_at field

        "approved_by": None,
        "approved_at": None,

        "executed_by": None,
        "executed_at": None,

        # ✅ AUDIT TRAIL
        "audit": [
            {"ts": now, "event": "created", "meta": {"by": "system"}}
        ],
    }

    DEMO_PROPOSALS.append(row)
    DEMO_PROPOSAL_BY_ID[pid] = row
    return row

def _case_or_404(case_id: int) -> Dict[str, Any]:
    row = DEMO_BY_ID.get(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return row

@router.get("/dashboard/api/automation/pending")
def list_pending_proposals():
    pending = [p for p in DEMO_PROPOSALS if p["status"] == "pending"]
    # newest first
    pending.sort(key=lambda x: x["created_at"], reverse=True)
    return {"items": pending, "count": len(pending)}

@router.get("/dashboard/api/activity")
def get_activity_feed(limit: int = 50):
    """Returns the live activity feed — newest first, max 200 entries."""
    items = list(reversed(ACTIVITY_FEED[-limit:]))
    return {"items": items, "count": len(items)}


@router.get("/dashboard/api/cases/{case_id}")
def get_case_detail(case_id: int, db=Depends(get_db)):
    log.info(f"case_detail: case_id={case_id}")

    # UI sentinel: no case selected
    if case_id == -1:
        raise HTTPException(status_code=400, detail="No case selected")
    
    # ----------------
    # DEMO case (negative IDs)
    # ----------------
    if case_id < 0:
        row = _case_or_404(case_id)

        proposals = [p for p in DEMO_PROPOSALS if p["case_id"] == case_id]
        proposals.sort(key=lambda x: x["created_at"], reverse=True)

        queue = row.get("queue")
        suggested = []

        if queue:
            nq = next_queue_for(queue)  # preview-only helper (no gating)
            if nq:
                # ML: predict outcome for the suggested action
                ml_sug = {}
                sug_conf, sug_safety = 0.86, 0.92
                try:
                    op = _ml_outcome(_AME_SITE)
                    if op.is_ready:
                        ml_sug = op.predict_for_action(
                            db, queue=queue, action_type="advance_queue",
                            from_queue=queue, to_queue=nq,
                            tenant_id="default", site_id=_AME_SITE,
                        )
                        if not ml_sug.get("cold_start"):
                            sug_conf = ml_sug["success_probability"]
                            sug_safety = ml_sug["safety_score"]
                except Exception:
                    pass
                suggested.append({
                    "type": "advance_queue",
                    "label": f"Advance case to next queue from '{queue}'",
                    "payload": {"from_queue": queue, "to_queue": nq},
                    "confidence": sug_conf,
                    "safety_score": sug_safety,
                    "ml_outcome": ml_sug,
                })

        ame = _ame_trust_info(db, queue) if queue else {}
        return {"case": row, "suggested_actions": suggested, "proposals": proposals, "ame": ame}

    # ----------------
    # DB workflow (positive IDs)
    # ----------------
    wf = workflow_service.get_workflow(db, case_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf_read = workflow_service.to_workflow_read(wf).model_dump()
    queue = wf_read.get("queue") or "data_entry"

    case_view = {
        "id": wf_read["id"],
        "name": wf_read.get("name") or f"Workflow #{wf_read['id']}",
        "state": wf_read.get("state", "unknown"),
        "queue": queue,
        "raw": wf_read,
    }

    suggested = []
    nq = next_queue_for(queue)
    if nq:
        suggested.append({
            "type": "advance_queue",
            "label": f"Advance workflow to next queue from '{queue}'",
            "payload": {"from_queue": queue, "to_queue": nq},
            "confidence": 0.80,
            "safety_score": 0.85,
        })

    ame = _ame_trust_info(db, queue)
    return {"case": case_view, "suggested_actions": suggested, "proposals": [], "ame": ame}

@router.post("/dashboard/api/cases/{case_id}/propose")
def propose_automation(
    case_id: int,
    action_type: str = Body(...),
    label: str = Body(...),
    payload: Dict[str, Any] = Body(default={}),
    confidence: float = Body(0.75),
    safety_score: float = Body(0.75),
    db=Depends(get_db),
):
    case = _case_or_404(case_id)
    queue = case.get("queue", "unknown")

    # --- ML: Outcome Predictor (replaces hardcoded confidence/safety if trained) ---
    ml_outcome = {}
    try:
        outcome_pred = _ml_outcome(_AME_SITE)
        if outcome_pred.is_ready:
            ml_outcome = outcome_pred.predict_for_action(
                db, queue=queue, action_type=action_type,
                from_queue=queue, to_queue=payload.get("to_queue", ""),
                tenant_id="default", site_id=_AME_SITE,
            )
            if not ml_outcome.get("cold_start"):
                confidence = ml_outcome["success_probability"]
                safety_score = ml_outcome["safety_score"]
    except Exception as exc:
        _ame_log.debug("ML outcome skipped: %s", exc)

    # --- AME Stage Resolution (proposals are how the system learns, always allowed) ---
    mode, meta = _ame_resolve_safe(db, queue=queue, action_type=action_type,
                                   confidence=confidence, safety=safety_score)

    action = {
        "type": action_type,
        "label": label,
        "payload": payload,
        "confidence": confidence,
        "safety_score": safety_score,
    }
    p = _mk_proposal(case_id, action)

    # --- ML: Decision Predictor (predict approval probability) ---
    ml_decision = {}
    try:
        decision_pred = _ml_decision(_AME_SITE)
        if decision_pred.is_ready:
            ml_decision = decision_pred.predict_for_proposal(
                db, queue=queue, action_type=action_type,
                confidence=confidence, safety=safety_score,
                tenant_id="default", site_id=_AME_SITE,
            )
    except Exception as exc:
        _ame_log.debug("ML decision skipped: %s", exc)

    # Store ML predictions on the proposal for later display
    p["ml_decision"] = ml_decision
    p["ml_outcome"] = ml_outcome

    # AME: log proposal creation
    _ame_safe(db, queue=queue, action_type=action_type,
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=p["id"],
              predicted_confidence=confidence,
              predicted_safety=safety_score,
              context={"case_id": case_id, "label": label})

    # Narrate the proposal
    case_name = case.get("name", f"Case #{case_id}")
    ml_note = ""
    if ml_decision and ml_decision.get("approve_probability") is not None and ml_decision.get("confidence_band") != "cold_start":
        ml_note = f" ML predicts {round(ml_decision['approve_probability'] * 100)}% chance of human approval."
    if ml_outcome and ml_outcome.get("success_probability") is not None and not ml_outcome.get("cold_start"):
        ml_note += f" Predicted success: {round(ml_outcome['success_probability'] * 100)}%."
    _narrate(
        f"PROPOSAL CREATED for {case_name}: {label}",
        "proposal",
        f"AME trust stage: {_short_stage(mode)}. Confidence: {round(confidence * 100)}%, Safety: {round(safety_score * 100)}%.{ml_note} "
        f"This proposal now awaits human review — a pharmacist or manager must approve or reject it before execution.",
    )

    return {
        "ok": True, "proposal": p, "ame_stage": mode,
        "ml_decision": ml_decision, "ml_outcome": ml_outcome,
    }

@router.post("/dashboard/api/automation/{proposal_id}/decide")
def decide_proposal(
    proposal_id: str,
    decision: str = Body(..., embed=True),
    decided_by: str = Body("human", embed=True),
    note: str = Body("", embed=True),
    db=Depends(get_db),
):
    try:
        p = DEMO_PROPOSAL_BY_ID.get(proposal_id)
        if not p:
            raise HTTPException(status_code=404, detail="Proposal not found")
        if p.get("status") != "pending":
            raise HTTPException(status_code=409, detail=f"Proposal already {p.get('status')}")
        if decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="decision must be approve|reject")

        # update status
        now = _now_iso()
        p["status"] = "approved" if decision == "approve" else "rejected"

        # FIXED: Consolidated attribution fields
        if decision == "approve":
            p["approved_by"] = decided_by
            p["approved_at"] = now
        else:
            p["rejected_by"] = decided_by
            p["rejected_at"] = now

        if "audit" not in p or not isinstance(p["audit"], list):
            p["audit"] = []

        p["audit"].append({
            "ts": now,
            "event": f"decision:{decision}",
            "meta": {"by": decided_by, "note": note},
        })

        # AME: log human decision on proposal
        action = p.get("action") or {}
        case = DEMO_BY_ID.get(p.get("case_id"))
        queue = (case or {}).get("queue", "unknown")
        case_name = (case or {}).get("name", f"Case #{p.get('case_id')}")
        _ame_safe(db, queue=queue, action_type=action.get("type", "advance_queue"),
                  event_type=AMEEventType.PROPOSAL_DECIDED.value,
                  proposal_id=proposal_id,
                  decision=decision, decision_by=decided_by,
                  decision_reason=note or decision)

        # Narrate the decision
        if decision == "approve":
            _narrate(
                f"APPROVED by {decided_by}: {action.get('label', 'action')} ({case_name})",
                "decision",
                f"Human approved this proposal. This approval is recorded by AME as a positive trust signal — "
                f"the system learns that actions like this in the '{queue}' queue are safe. "
                f"The proposal can now be executed.",
            )
        else:
            _narrate(
                f"REJECTED by {decided_by}: {action.get('label', 'action')} ({case_name})",
                "decision",
                f"Human rejected this proposal. This rejection teaches AME that the system's suggestion was wrong — "
                f"too many rejections will slow or prevent trust stage promotion for the '{queue}' queue.",
            )

        return {"ok": True, "proposal": p}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("decide_proposal crashed")
        raise HTTPException(status_code=500, detail=f"decide_proposal error: {type(e).__name__}: {e}")

@router.post("/dashboard/api/automation/{proposal_id}/execute")
def execute_proposal(
    proposal_id: str,
    executed_by: str = Body("system", embed=True),
    db=Depends(get_db),
):
    p = DEMO_PROPOSAL_BY_ID.get(proposal_id)
    if not p:
        raise HTTPException(status_code=404, detail="Proposal not found")
    if p["status"] != "approved":
        raise HTTPException(status_code=409, detail="Only approved proposals can be executed")

    case = _case_or_404(p["case_id"])
    action = p.get("action") or {}
    act_type = action.get("type", "advance_queue")
    queue = case.get("queue", "unknown")

    # --- AME Stage Enforcement ---
    mode, meta = _ame_resolve_safe(db, queue=queue, action_type=act_type,
                                   confidence=action.get("confidence", 0.75),
                                   safety=action.get("safety_score", 0.85))
    skip_governance = mode in ("guarded_auto", "conditional_auto", "full_auto")
    needs_rollback = mode in ("guarded_auto", "conditional_auto")

    # ensure audit exists
    if "audit" not in p or not isinstance(p["audit"], list):
        p["audit"] = []

    now = _now_iso()

    # Demo execution: handle "advance_queue"
    if action.get("type") == "advance_queue":
        cur = case.get("queue") or "data_entry"

        payload = action.get("payload") or {}
        to_q = payload.get("to_queue") or next_queue_for(cur)

        if not to_q:
            p["audit"].append({
                "ts": now,
                "event": "executed:noop",
                "meta": {"reason": f"no transition from {cur}"},
            })
        else:
            gk = gate_for_transition(cur, to_q)

            if skip_governance and gk:
                # AME trust level allows bypassing governance gate
                p["audit"].append({
                    "ts": now,
                    "event": "governance_bypassed",
                    "meta": {"reason": f"AME stage {mode}", "gate": gk,
                             "trust": meta.get("trust", 0.0)},
                })
            elif gk:
                try:
                    governance.require_authorized(gk)
                except Exception as e:
                    p["audit"].append({
                        "ts": now,
                        "event": "execution_failed",
                        "meta": {"reason": f"governance check failed: {str(e)}"},
                    })
                    raise HTTPException(status_code=403, detail=f"Not authorized: {gk}")

            before_state = {"queue": cur}
            case["queue"] = to_q
            case["status"] = "demo"
            p["audit"].append({
                "ts": now,
                "event": "executed:advance_queue",
                "meta": {"from": cur, "to": case["queue"], "ame_stage": mode},
            })

            # GUARDED/CONDITIONAL: create execution record with rollback window
            if needs_rollback:
                try:
                    exe = _ame_create_execution(
                        db, site_id=_AME_SITE, queue=cur, action_type=act_type,
                        proposal_id=proposal_id,
                        before_state=before_state,
                        after_state={"queue": to_q},
                        guarded=True,
                    )
                    db.commit()
                    p["audit"].append({
                        "ts": now,
                        "event": "rollback_window_created",
                        "meta": {"execution_id": exe.id,
                                 "reversible_until": str(exe.reversible_until)},
                    })
                except Exception as exc:
                    _ame_log.warning("AME execution record skipped: %s", exc)
    else:
        p["audit"].append({
            "ts": now,
            "event": "executed:noop",
            "meta": {"reason": "unknown action type"},
        })

    # IMPORTANT: mark executed regardless of branch
    p["status"] = "executed"
    p["executed_at"] = now
    p["executed_by"] = executed_by
    p["audit"].append({
        "ts": now,
        "event": "executed",
        "meta": {"by": executed_by, "ame_stage": mode},
    })

    # AME: log execution + successful outcome
    _ame_safe(db, queue=queue, action_type=act_type,
              event_type=AMEEventType.EXECUTED.value,
              proposal_id=proposal_id)
    _ame_safe(db, queue=queue, action_type=act_type,
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=proposal_id,
              outcome_success=True, observed_time_saved_sec=30.0)

    ame = _ame_trust_info(db, queue, act_type)

    # Narrate the execution
    case_name = case.get("name", f"Case #{case.get('id')}")
    act_label = action.get("label", action.get("type", "action"))
    payload = action.get("payload") or {}
    from_q = payload.get("from_queue", queue)
    to_q = payload.get("to_queue", case.get("queue", "?"))
    if skip_governance:
        gov_note = (
            f"Governance gate BYPASSED — AME trust stage '{_short_stage(mode)}' "
            f"(trust: {round(meta.get('trust', 0) * 100)}%) allows automated execution."
        )
        if needs_rollback:
            gov_note += " A 15-minute rollback window was created — a human can still undo this action."
    else:
        gov_note = "Governance gate was checked and authorized by a human operator before execution."
    _narrate(
        f"EXECUTED: {case_name} moved {from_q} \u2192 {to_q}",
        "execution",
        f"{gov_note} AME recorded this as a successful outcome (+30s saved). "
        f"Each successful execution builds trust toward higher automation stages.",
        meta={"ame_stage": mode, "trust": ame.get("trust", 0)},
    )

    # Check if trust stage might have changed
    new_stage = ame.get("stage", "observe")
    if new_stage != mode:
        _narrate(
            f"TRUST STAGE CHANGED: '{queue}' queue advanced to {_short_stage(new_stage)}",
            "trust",
            _stage_label(new_stage),
        )

    return {"ok": True, "proposal": p, "case": case, "ame": ame, "ame_stage": mode}

@router.get("/dashboard/api/scenarios")
def list_demo_scenarios():
    """
    Returns available demo scenarios for the dashboard dropdown.
    """
    items = []
    for sid, s in DEMO_SCENARIOS.items():
        items.append({
            "id": sid,
            "label": s.get("label", sid),
            "description": s.get("description", ""),
            "start_queue": s.get("start_queue", "inbound_comms"),
        })
    return {"scenarios": items}

# =============================
# API: Simulate repetition (events + optional task repetition)
# =============================

def _demo_repeat_tasks(row: dict, copies: int = 1):
    """
    Adds repeated tasks to show repetition/volume. Uses the first task as a template if present.
    """
    tasks = row["raw"].setdefault("tasks", [])
    if not tasks:
        tasks.append({"name": "Demo task", "assigned_to": "—", "state": "open"})

    template = tasks[0]
    base_name = template.get("name", "Demo task")
    for i in range(copies):
        tasks.append({
            "name": f"{base_name} (repeat {len(tasks)})",
            "assigned_to": template.get("assigned_to", "—"),
            "state": template.get("state", "open"),
        })

@router.get("/dashboard/api/ml-stats")
def dashboard_ml_stats(db=Depends(get_db)):
    """ML model status + learning metrics + ROI for the dashboard UI."""
    from src.enterprise.ame.models import AMEEvent as _Ev
    from src.enterprise.ame.ml.trainer import AMETrainer

    total = db.query(_Ev).filter(_Ev.site_id == _AME_SITE, _Ev.deleted_at.is_(None)).count()
    proposals = db.query(_Ev).filter(
        _Ev.site_id == _AME_SITE, _Ev.event_type == "proposal_created", _Ev.deleted_at.is_(None),
    ).count()

    decisions = db.query(_Ev).filter(
        _Ev.site_id == _AME_SITE, _Ev.event_type == "proposal_decided", _Ev.deleted_at.is_(None),
    ).all()
    approved = sum(1 for d in decisions if d.decision == "approve")
    rejected = sum(1 for d in decisions if d.decision == "reject")

    outcomes = db.query(_Ev).filter(
        _Ev.site_id == _AME_SITE, _Ev.event_type == "outcome", _Ev.deleted_at.is_(None),
    ).all()
    successes = sum(1 for o in outcomes if o.outcome_success)
    time_saved = sum(float(o.observed_time_saved_sec or 0) for o in outcomes)

    # Count governance-bypassed executions (auto-approved)
    auto_approved = sum(
        1 for p in DEMO_PROPOSALS
        if any(a.get("event") == "governance_bypassed" for a in (p.get("audit") or []))
    )

    trainer = AMETrainer()
    models = trainer.get_status()

    return {
        "events": {"total": total, "proposals": proposals, "approved": approved,
                    "rejected": rejected, "successes": successes},
        "roi": {"auto_approved": auto_approved,
                "human_reviewed": max(0, approved + rejected - auto_approved),
                "time_saved_sec": round(time_saved, 1),
                "time_saved_min": round(time_saved / 60, 1)},
        "models": models,
    }


@router.post("/dashboard/api/seed-ml")
def seed_ml_training_data(db=Depends(get_db)):
    """
    Seed realistic ML training data: a mix of approvals, rejections, successes,
    and failures across all queues. Generates enough events to train all 3 models.
    """
    import random
    from datetime import timedelta
    from src.enterprise.ame.models import AMEEvent

    random.seed(42)
    now = datetime.utcnow()
    queues = list(TRANSITIONS.keys())
    created = 0

    # Generate 80 events spread over 24 hours (enough for anomaly windows)
    n_events = 80
    for i in range(n_events):
        q = queues[i % len(queues)]
        ts = now - timedelta(hours=24) + timedelta(minutes=i * 18)  # ~18 min apart over 24h
        pid = f"seed-{uuid.uuid4().hex[:8]}"

        # Proposal
        conf = 0.65 + random.random() * 0.30
        safety = 0.70 + random.random() * 0.25
        db.add(AMEEvent(
            tenant_id="default", site_id=_AME_SITE, queue=q,
            action_type="advance_queue", event_type="proposal_created",
            proposal_id=pid, predicted_confidence=round(conf, 3),
            predicted_safety=round(safety, 3),
            predicted_time_saved_sec=20 + random.random() * 20,
            ts=ts,
        ))

        # Decision: 70% approve, 30% reject (more rejections for training)
        decision = "approve" if random.random() < 0.70 else "reject"
        db.add(AMEEvent(
            tenant_id="default", site_id=_AME_SITE, queue=q,
            action_type="advance_queue", event_type="proposal_decided",
            proposal_id=pid, decision=decision, decision_by="demo_pharmacist",
            decision_reason="seeded decision",
            ts=ts + timedelta(seconds=15),
        ))

        # Outcome (only for approved): 80% success, 20% failure
        if decision == "approve":
            success = random.random() < 0.80
            db.add(AMEEvent(
                tenant_id="default", site_id=_AME_SITE, queue=q,
                action_type="advance_queue", event_type="outcome",
                proposal_id=pid, outcome_success=success,
                observed_error=not success,
                observed_time_saved_sec=25 + random.random() * 15 if success else 0,
                ts=ts + timedelta(seconds=30),
            ))

        created += 1

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    # Auto-train after seeding
    from src.enterprise.ame.ml.trainer import AMETrainer
    trainer = AMETrainer()
    results = trainer.train_all(db)
    _ml_clear_cache()

    # Narrate ML training
    parts = []
    if results.get("decision", {}).get("trained"):
        acc = round(results["decision"].get("metrics", {}).get("accuracy", 0) * 100)
        parts.append(f"Decision Predictor ({acc}% accuracy)")
    if results.get("outcome", {}).get("trained"):
        acc = round(results["outcome"].get("metrics", {}).get("accuracy", 0) * 100)
        parts.append(f"Outcome Predictor ({acc}% accuracy)")
    if results.get("anomaly", {}).get("trained"):
        parts.append("Anomaly Detector")
    _narrate(
        f"ML MODELS TRAINED on {created} synthetic events",
        "ml",
        f"Models now active: {', '.join(parts) if parts else 'none (insufficient data)'}. "
        f"The Decision Predictor forecasts human approval likelihood. "
        f"The Outcome Predictor estimates action success probability. "
        f"These predictions appear on proposals and inform auto-step decisions.",
    )

    return {"ok": True, "events_created": created, "training_results": results}


@router.post("/dashboard/api/reset")
def reset_dashboard(db=Depends(get_db)):
    """
    Full reset: clears demo cases, proposals, AME trust data, and governance gates.
    Returns the dashboard to a clean-slate state for a fresh demo.
    """
    global DEMO_ROWS, DEMO_BY_ID, DEMO_PROPOSALS, DEMO_PROPOSAL_BY_ID

    # 1. Clear in-memory demo data
    DEMO_ROWS = []
    DEMO_BY_ID = {}
    DEMO_PROPOSALS = []
    DEMO_PROPOSAL_BY_ID = {}

    # 2. Clear AME trust tables
    from src.enterprise.ame.models import AMETrustScope, AMEEvent, AMEExecution
    try:
        db.query(AMEExecution).delete()
        db.query(AMEEvent).delete()
        db.query(AMETrustScope).delete()
        db.commit()
        log.info("AME trust data cleared")
    except Exception as exc:
        db.rollback()
        log.warning("AME reset failed: %s", exc)

    # 3. Reset governance gates
    try:
        governance.reset_all_gates(actor="dashboard_reset", note="Dashboard full reset")
        log.info("Governance gates reset")
    except Exception as exc:
        log.warning("Governance reset failed: %s", exc)

    # 4. Clear ML model cache and persisted model files
    try:
        _ml_clear_cache()
        import shutil
        model_dir = os.path.join("data", "models")
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
            log.info("ML model files removed: %s", model_dir)
        log.info("ML model cache cleared")
    except Exception as exc:
        log.warning("ML cache clear failed: %s", exc)

    # 5. Clear activity feed
    ACTIVITY_FEED.clear()
    _narrate(
        "Dashboard reset to clean slate.",
        "info",
        "All demo cases, proposals, AME trust data, governance gates, and ML models cleared. "
        "Ready for a fresh demo — seed scenarios to begin.",
    )

    return {"ok": True, "message": "Dashboard reset to clean slate"}

@router.post("/dashboard/api/seed")
def seed_demo_cases(
    scenario_id: str = Body("happy_path", embed=True),
    seed_all: bool = Body(False, embed=True),
):
    global DEMO_ROWS, DEMO_BY_ID, DEMO_PROPOSALS, DEMO_PROPOSAL_BY_ID
    DEMO_ROWS = []
    DEMO_BY_ID = {}
    DEMO_PROPOSALS = []
    DEMO_PROPOSAL_BY_ID = {}

    def _mk_case(sid: str, idx: int):
        s = DEMO_SCENARIOS.get(sid, DEMO_SCENARIOS["happy_path"])
        demo_id = -(len(DEMO_ROWS) + 1)

        # Each scenario starts at its own queue for visual variety
        start_queue = s.get("start_queue", "inbound_comms")
        scenario_label = s.get("label", sid)

        # Build scenario-specific events that tell the story
        events = [
            {"event_type": "case_seeded", "payload": {"scenario_id": sid, "label": scenario_label, "description": s.get("description", "")}},
            {"event_type": "queue_changed", "payload": {"from": "none", "to": start_queue}},
            {"event_type": "insurance_adjudicated", "payload": {"payer": "AutoPayer", "result": s.get("insurance_result", "accepted")}},
        ]

        # Add scenario-specific context events
        if s.get("reject_reason"):
            events.append({"event_type": "insurance_rejection_detail", "payload": {"reason": s["reject_reason"]}})
        if s.get("pa"):
            events.append({"event_type": "prior_auth_initiated", "payload": {"eta_days": s["pa"]["eta_days"], "status": "pending"}})
        if s.get("prescriber_request"):
            events.append({"event_type": "prescriber_request_sent", "payload": s["prescriber_request"]})
        if s.get("discount_card"):
            events.append({"event_type": "discount_card_applied", "payload": s["discount_card"]})
        if s.get("patient_message"):
            events.append({"event_type": "patient_notification_queued", "payload": s["patient_message"]})

        # Build scenario-specific tasks
        tasks = [{"name": "Enter NPI + patient DOB", "assigned_to": "—", "state": "open"}]
        if not s.get("refills_ok"):
            tasks.append({"name": "Request refill authorization from prescriber", "assigned_to": "Store Tech", "state": "pending"})
        if s.get("insurance_result") == "pa_required":
            tasks.append({"name": "Monitor prior authorization status", "assigned_to": "System", "state": "pending"})
        if s.get("insurance_result") == "rejected":
            tasks.append({"name": "Contact patient re: insurance update", "assigned_to": "Store Tech", "state": "pending"})

        raw = {
            "id": demo_id,
            "name": f"Kroger \u2022 RX-{1000 + idx} (Demo)",
            "state": "ACTIVE",
            "scenario_id": sid,
            "scenario_label": scenario_label,
            "tasks": tasks,
            "events": events,
        }

        row = {
            "id": demo_id,
            "name": raw["name"],
            "state": raw["state"],
            "queue": start_queue,
            "scenario_id": sid,
            "scenario_label": scenario_label,
            "insurance": f"AutoPayer: {s.get('insurance_result','accepted')}",
            "tasks": len(raw["tasks"]),
            "events": len(raw["events"]),
            "is_kroger": True,
            "raw": raw,
        }

        DEMO_ROWS.append(row)
        DEMO_BY_ID[demo_id] = row
        return row

    if seed_all:
        for i, sid in enumerate(DEMO_SCENARIOS.keys(), start=1):
            _mk_case(sid, i)
        _narrate(
            f"Seeded {len(DEMO_ROWS)} demo cases across the pipeline.",
            "info",
            "Each scenario starts at a different queue to show the full pharmacy workflow. "
            "Cases are governed by authorization gates — toggle gates to allow progression. "
            "AME trust starts at OBSERVE: every action requires human approval.",
        )
        return {"ok": True, "count": len(DEMO_ROWS)}
    else:
        row = _mk_case(scenario_id, 1)
        s = DEMO_SCENARIOS.get(scenario_id, {})
        _narrate(
            f"Seeded: {s.get('label', scenario_id)}",
            "info",
            s.get("seed_narrative", s.get("description", "")),
        )
        return {"ok": True, "count": 1, "id": row["id"]}

@router.get("/dashboard/api/workflows")
def dashboard_list_workflows(db=Depends(get_db)):
    rows = []

    # 1) DB workflows (if any)
    try:
        wfs = workflow_service.list_workflows(db)
        for wf in wfs:
            wf_read = workflow_service.to_workflow_read(wf).model_dump()
            rows.append({
                "id": wf_read["id"],
                "name": wf_read.get("name") or f"Workflow #{wf_read['id']}",
                "state": wf_read.get("state", "unknown"),
                "queue": wf_read.get("queue") or "data_entry",  # FIXED: Extract from workflow
                "insurance": "—",
                "tasks": len(wf_read.get("tasks") or []),
                "events": len(wf_read.get("events") or []),
                "is_kroger": "kroger" in (wf_read.get("name","").lower()),
                "raw": wf_read,
            })
    except Exception:
        # keep dashboard working even if DB list fails
        pass

    # 2) Demo workflows (seeded)
    rows.extend(DEMO_ROWS)

    return {"workflows": rows}

@router.post("/dashboard/api/simulate")
def simulate_repetition(
    workflow_id: int = Body(..., embed=True),
    cycles: int = Body(5, embed=True),
    repeat_tasks: bool = Body(False, embed=True),
    db=Depends(get_db),
):
    """
    Simulates repetition:
    - Demo workflows (negative ids): appends demo_repetition_tick events; optionally repeats tasks.
    - Real DB workflows: emits demo_repetition_tick events via EventCreate (tasks repetition is demo-only).
    """
    if cycles < 1:
        return {"ok": True, "did": 0}

    # DEMO workflow
    if workflow_id < 0:
        row = DEMO_BY_ID.get(workflow_id)
        if not row:
            raise HTTPException(status_code=404, detail="Demo workflow not found")

        for i in range(cycles):
            row["raw"]["events"].append({
                "event_type": "demo_repetition_tick",
                "payload": {"tick": i + 1, "note": "simulated repetition"},
            })

        if repeat_tasks:
            _demo_repeat_tasks(row, copies=cycles)

        row["events"] = len(row["raw"].get("events") or [])
        row["tasks"] = len(row["raw"].get("tasks") or [])
        return {"ok": True, "did": cycles, "events": row["events"], "tasks": row["tasks"]}

    # REAL DB workflow: only add events (tasks repetition depends on your DB task model)
    wf = workflow_service.get_workflow(db, workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    for i in range(cycles):
        workflow_service.add_event(
            db,
            workflow_id,
            EventCreate(event_type="demo_repetition_tick", payload={"tick": i + 1, "note": "simulated repetition"})
        )

    wf2 = workflow_service.get_workflow(db, workflow_id)
    return {"ok": True, "did": cycles, "workflow": workflow_service.to_workflow_read(wf2).model_dump()}

@router.post("/dashboard/api/auto-step")
def dashboard_auto_step(
    workflow_id: int = Body(..., embed=True),
    db=Depends(get_db),
):
    """
    AME-governed auto-step for demo + DB workflows.

    Stage enforcement:
      OBSERVE / PROPOSE  → governance gate required (human toggles)
      GUARDED_AUTO       → governance bypassed, rollback window created
      CONDITIONAL_AUTO   → governance bypassed (thresholds already validated)
      FULL_AUTO          → governance bypassed, immediate execution
    """

    def _step_with_enforcement(cur_queue):
        """Resolve AME stage and step the queue accordingly.
        Returns (next_queue, gate_key, ame_stage, ame_meta, execution_record_or_None).
        """
        t = TRANSITIONS.get(cur_queue)
        if not t:
            return cur_queue, None, "observe", {}, None

        to_q = t["to"]
        gk = t["gate"]

        # ML: Use Outcome Predictor for confidence/safety (replaces hardcoded 0.86/0.92)
        ml_conf, ml_safety = 0.86, 0.92
        try:
            op = _ml_outcome(_AME_SITE)
            if op.is_ready:
                ml_pred = op.predict_for_action(
                    db, queue=cur_queue, action_type="advance_queue",
                    from_queue=cur_queue, to_queue=to_q,
                    tenant_id="default", site_id=_AME_SITE,
                )
                if not ml_pred.get("cold_start"):
                    ml_conf = ml_pred["success_probability"]
                    ml_safety = ml_pred["safety_score"]
        except Exception as exc:
            _ame_log.debug("ML outcome in auto-step skipped: %s", exc)

        mode, meta = _ame_resolve_safe(
            db, queue=cur_queue, action_type="advance_queue",
            confidence=ml_conf, safety=ml_safety,
        )

        exe_record = None
        if mode in ("guarded_auto", "conditional_auto", "full_auto"):
            # Trust level allows bypassing governance gate
            log.info("AME stage %s: bypassing governance gate %s for %s→%s",
                     mode, gk, cur_queue, to_q)

            # GUARDED/CONDITIONAL: create execution record with rollback window
            if mode in ("guarded_auto", "conditional_auto"):
                try:
                    exe_record = _ame_create_execution(
                        db, site_id=_AME_SITE, queue=cur_queue,
                        action_type="advance_queue", proposal_id=None,
                        before_state={"queue": cur_queue},
                        after_state={"queue": to_q},
                        guarded=True,
                    )
                    db.commit()
                except Exception as exc:
                    _ame_log.warning("AME execution record skipped: %s", exc)
        else:
            # OBSERVE / PROPOSE: governance gate required
            governance.require_authorized(gk)

        return to_q, gk, mode, meta, exe_record

    # ----------------
    # DEMO case (negative IDs)
    # ----------------
    if workflow_id < 0:
        row = DEMO_BY_ID.get(workflow_id)
        if not row:
            raise HTTPException(status_code=404, detail="Demo workflow not found")

        cur = row.get("queue") or "data_entry"
        nxt, gk, mode, meta, exe = _step_with_enforcement(cur)

        if nxt == cur:
            return {"ok": True, "note": f"No rule for queue '{cur}'", "queue": cur,
                    "ame_stage": mode}

        row["raw"]["events"].append({"event_type": "auto_step",
                                     "payload": {"from": cur, "to": nxt, "ame_stage": mode}})
        row["raw"]["events"].append({"event_type": "queue_changed",
                                     "payload": {"from": cur, "to": nxt}})
        row["queue"] = nxt
        row["events"] = len(row["raw"].get("events") or [])

        # AME: log auto-step as a complete trust cycle (use ML-predicted confidence/safety)
        pid = f"dash-{uuid.uuid4().hex[:8]}"
        _step_conf = meta.get("model_confidence", 0.86)
        _step_safety = meta.get("model_safety", 0.92)
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_CREATED.value,
                  proposal_id=pid, predicted_confidence=_step_conf, predicted_safety=_step_safety,
                  predicted_time_saved_sec=30.0,
                  context={"transition": f"{cur}→{nxt}", "workflow_id": workflow_id,
                           "ame_stage": mode})
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_DECIDED.value,
                  proposal_id=pid, decision="approve", decision_by="auto_step",
                  decision_reason=f"AME {mode}: auto-step {cur}→{nxt}")
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.OUTCOME.value,
                  proposal_id=pid, outcome_success=True, observed_time_saved_sec=30.0)

        ame = _ame_trust_info(db, cur)

        # Narrate auto-step
        case_name = row.get("name", f"Case #{workflow_id}")
        trust_pct = round(ame.get("trust", 0) * 100)
        if mode in ("guarded_auto", "conditional_auto", "full_auto"):
            gov_msg = f"AME trust stage '{_short_stage(mode)}' ({trust_pct}%) allowed automated execution — governance gate bypassed."
            if exe:
                gov_msg += f" Rollback window active: a human can undo this within 15 minutes."
        else:
            gov_msg = f"AME stage is '{_short_stage(mode)}' — governance gate was required and checked before advancement."
        _narrate(
            f"AUTO-STEP: {case_name} moved {cur} \u2192 {nxt}",
            "execution",
            f"{gov_msg} AME logged a complete trust cycle (propose \u2192 approve \u2192 outcome). "
            f"Trust score: {trust_pct}%. Each successful auto-step builds toward higher automation.",
            meta={"ame_stage": mode, "trust": ame.get("trust", 0)},
        )

        # Check for stage change
        new_stage = ame.get("stage", "observe")
        if new_stage != mode:
            _narrate(
                f"TRUST STAGE CHANGED: '{cur}' queue is now {_short_stage(new_stage)}!",
                "trust",
                _stage_label(new_stage),
            )

        result = {"ok": True, "from": cur, "to": nxt, "ame": ame, "ame_stage": mode}
        if exe:
            result["rollback_until"] = str(exe.reversible_until)
            result["execution_id"] = exe.id
        return result

    # ----------------
    # DB workflow (positive IDs): append events only (pilot-safe)
    # ----------------
    wf = workflow_service.get_workflow(db, workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf_read = workflow_service.to_workflow_read(wf).model_dump()
    cur = wf_read.get("queue") or "data_entry"
    nxt, gk, mode, meta, exe = _step_with_enforcement(cur)

    workflow_service.add_event(db, workflow_id, EventCreate(
        event_type="auto_step", payload={"from": cur, "to": nxt, "ame_stage": mode}))
    workflow_service.add_event(db, workflow_id, EventCreate(
        event_type="queue_changed", payload={"from": cur, "to": nxt}))

    # AME: log auto-step trust cycle for DB workflows (use ML-predicted values)
    pid = f"dash-{uuid.uuid4().hex[:8]}"
    _step_conf = meta.get("model_confidence", 0.86)
    _step_safety = meta.get("model_safety", 0.92)
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=_step_conf, predicted_safety=_step_safety,
              predicted_time_saved_sec=30.0,
              context={"transition": f"{cur}→{nxt}", "workflow_id": workflow_id,
                       "ame_stage": mode})
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=pid, decision="approve", decision_by="auto_step",
              decision_reason=f"AME {mode}: auto-step {cur}→{nxt}")
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=pid, outcome_success=True, observed_time_saved_sec=30.0)

    ame = _ame_trust_info(db, cur)
    wf2 = workflow_service.get_workflow(db, workflow_id)
    result = {"ok": True, "workflow": workflow_service.to_workflow_read(wf2).model_dump(),
              "from": cur, "to": nxt, "ame": ame, "ame_stage": mode}
    if exe:
        result["rollback_until"] = str(exe.reversible_until)
        result["execution_id"] = exe.id
    return result

# =============================
# UI: /dashboard (UPDATED)
# =============================

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Bladnir Tech — PactGate™</title>
  <style>
    :root{
      --bg:#0b0f14;--card:#111823;--line:#1f2a36;--text:#eaeaea;--muted:#9bb0c5;
      --btn:#ffffff;--btnText:#0b0f14;--pill:#0d131c;--accent:#3b82f6;
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:'Inter',system-ui,-apple-system,Arial,sans-serif;background:var(--bg);color:var(--text);font-size:14px}

    /* --- Header --- */
    header{display:flex;align-items:center;gap:12px;padding:11px 20px;border-bottom:1px solid var(--line);background:#080b10}
    header b{font-size:15px;letter-spacing:.3px}
    .muted{color:var(--muted);font-size:12px}

    /* --- Controls bar --- */
    .controls{display:flex;align-items:center;gap:10px;padding:10px 20px;border-bottom:1px solid var(--line);background:var(--card);flex-wrap:wrap}

    /* --- Shared UI --- */
    button{cursor:pointer;border-radius:8px;padding:7px 12px;border:1px solid var(--line);background:transparent;color:var(--text);font-size:12px;white-space:nowrap}
    button:hover{background:rgba(255,255,255,.05)}
    button.primary{background:var(--btn);color:var(--btnText);border-color:var(--btn);font-weight:700}
    button.primary:hover{opacity:.9}
    input,select{padding:7px 10px;border-radius:8px;border:1px solid var(--line);background:var(--pill);color:var(--text);font-size:12px}
    select{min-width:160px}
    .pill{background:var(--pill);border:1px solid var(--line);padding:4px 10px;border-radius:999px;font-size:11px;color:var(--muted)}
    .item{border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--pill);margin-bottom:6px;font-size:12px}

    /* --- Pipeline --- */
    .pipeline{display:flex;align-items:flex-start;padding:16px 20px;overflow-x:auto;gap:0}
    .pipe-arrow{display:flex;align-items:center;justify-content:center;padding:0 2px;color:var(--muted);opacity:.45;padding-top:44px;flex-shrink:0}
    .pipe-arrow svg{width:22px;height:22px}
    .pipe-col{flex:1;min-width:155px;max-width:300px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
    .pipe-header{padding:10px 12px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.02)}
    .pipe-title{font-weight:700;font-size:12px;margin-bottom:1px}
    .pipe-badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-top:5px}
    .pipe-badge[data-stage="observe"]{background:rgba(107,114,128,.25);color:#9ca3af}
    .pipe-badge[data-stage="propose"]{background:rgba(59,130,246,.2);color:#60a5fa}
    .pipe-badge[data-stage="guarded_auto"]{background:rgba(245,158,11,.2);color:#fbbf24}
    .pipe-badge[data-stage="conditional_auto"]{background:rgba(249,115,22,.2);color:#fb923c}
    .pipe-badge[data-stage="full_auto"]{background:rgba(34,197,94,.2);color:#4ade80}
    .pipe-badge[data-stage="-"]{background:rgba(107,114,128,.12);color:#6b7280}
    .pipe-trust{height:3px;background:rgba(255,255,255,.06);border-radius:2px;margin-top:6px;overflow:hidden}
    .pipe-trust-fill{height:100%;border-radius:2px;transition:width .4s ease}
    .pipe-body{padding:8px;max-height:260px;overflow-y:auto}
    .pipe-body:empty::after{content:'Empty';color:var(--muted);font-size:11px;display:block;text-align:center;padding:18px 0;opacity:.6}
    .pipe-card{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--pill);margin-bottom:5px;cursor:pointer;transition:border-color .15s,background .15s}
    .pipe-card:hover{border-color:rgba(255,255,255,.15);background:rgba(255,255,255,.03)}
    .pipe-card.active{border-color:var(--accent);background:rgba(59,130,246,.08)}
    .pipe-card-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .pipe-card-ins{font-size:10px;color:var(--muted);margin-top:2px}

    /* --- Detail panel --- */
    .detail{padding:0 20px 20px;display:none}
    .detail.open{display:block}
    .detail-bar{display:flex;align-items:center;gap:10px;padding:12px 0;flex-wrap:wrap}
    .detail-title{font-weight:700;font-size:14px}
    .detail-pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:2px}
    .detail-actions{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap;align-items:center}
    .auth-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:8px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:6px 0 10px}
    .auth-row label{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted);cursor:pointer}
    .auth-row input[type=checkbox]{accent-color:var(--accent)}
    .detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .detail-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
    .detail-card b{font-size:12px;display:block;margin-bottom:8px}
    pre{white-space:pre-wrap;word-break:break-word;background:#080a0e;border-radius:10px;padding:12px;border:1px solid var(--line);color:#a3e635;font-size:11px;margin-top:8px;max-height:260px;overflow:auto}

    /* --- ML Intelligence Strip --- */
    .ml-strip{padding:12px 20px;border-bottom:1px solid var(--line);background:rgba(8,11,16,.9)}
    .ml-strip-header{display:flex;align-items:center;gap:10px;margin-bottom:10px}
    .ml-strip-title{font-weight:700;font-size:12px;color:#60a5fa;text-transform:uppercase;letter-spacing:.5px}
    .ml-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1.5fr;gap:10px}
    .ml-card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;position:relative}
    .ml-card-label{font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
    .ml-card-value{font-size:20px;font-weight:700;margin-top:3px;line-height:1.1}
    .ml-card-sub{font-size:11px;color:var(--muted);margin-top:4px}
    .ml-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle}
    .ml-dot.off{background:#6b7280}.ml-dot.on{background:#4ade80}.ml-dot.warn{background:#fbbf24}
    .roi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:6px}
    .roi-stat{text-align:center}
    .roi-num{display:block;font-size:18px;font-weight:700;color:#fff;line-height:1.2}
    .roi-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.3px}

    /* --- ML prediction in proposals --- */
    .ml-pred{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);border-radius:8px;padding:8px 10px;margin-top:6px;font-size:11px}
    .ml-pred-row{display:flex;justify-content:space-between;align-items:center;gap:8px}
    .ml-pred-bar{height:4px;background:rgba(255,255,255,.06);border-radius:2px;flex:1;max-width:80px;overflow:hidden}
    .ml-pred-fill{height:100%;border-radius:2px}
    .ml-factors{color:var(--muted);font-size:10px;margin-top:3px}

    /* --- Anomaly alert --- */
    .anomaly-banner{padding:8px 20px;background:rgba(248,113,113,.1);border-bottom:1px solid rgba(248,113,113,.25);color:#f87171;font-size:12px;display:none;align-items:center;gap:8px}
    .anomaly-banner.show{display:flex}

    /* --- Activity Feed --- */
    .feed-strip{padding:0 20px;border-bottom:1px solid var(--line);background:rgba(8,11,16,.85);max-height:0;overflow:hidden;transition:max-height .35s ease,padding .35s ease}
    .feed-strip.open{max-height:220px;padding:10px 20px}
    .feed-toggle{display:flex;align-items:center;gap:8px;padding:6px 20px;border-bottom:1px solid var(--line);background:rgba(8,11,16,.6);cursor:pointer;user-select:none}
    .feed-toggle:hover{background:rgba(255,255,255,.03)}
    .feed-toggle-title{font-size:11px;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.5px}
    .feed-toggle-arrow{color:var(--muted);font-size:12px;transition:transform .25s}
    .feed-toggle-arrow.open{transform:rotate(180deg)}
    .feed-badge{background:rgba(167,139,250,.15);color:#a78bfa;font-size:10px;padding:1px 7px;border-radius:999px;font-weight:600}
    .feed-list{max-height:180px;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
    .feed-item{font-size:11.5px;line-height:1.45;padding:5px 8px;border-radius:6px;border-left:3px solid transparent;background:rgba(255,255,255,.02)}
    .feed-item[data-cat="proposal"]{border-left-color:#3b82f6}
    .feed-item[data-cat="decision"]{border-left-color:#a78bfa}
    .feed-item[data-cat="execution"]{border-left-color:#4ade80}
    .feed-item[data-cat="trust"]{border-left-color:#fbbf24}
    .feed-item[data-cat="gate"]{border-left-color:#f97316}
    .feed-item[data-cat="ml"]{border-left-color:#06b6d4}
    .feed-item[data-cat="warning"]{border-left-color:#f87171}
    .feed-item[data-cat="info"]{border-left-color:#6b7280}
    .feed-msg{font-weight:600;color:#eaeaea}
    .feed-detail{color:var(--muted);font-size:10.5px;margin-top:1px}
    .feed-ts{color:#4b5563;font-size:9.5px;float:right;margin-left:8px}

    /* --- Scenario description --- */
    .scenario-desc{font-size:11px;color:var(--muted);padding:6px 20px;border-bottom:1px solid var(--line);background:rgba(8,11,16,.5);display:none;line-height:1.5}
    .scenario-desc.show{display:block}

    /* --- Scenario label on case card --- */
    .pipe-card-scenario{font-size:9px;color:#a78bfa;margin-top:2px;font-weight:600;text-transform:uppercase;letter-spacing:.3px}

    /* --- Stage tooltip --- */
    .stage-tip{position:relative;cursor:help}
    .stage-tip:hover .stage-tip-text{opacity:1;pointer-events:auto;transform:translateY(0)}
    .stage-tip-text{position:absolute;bottom:calc(100% + 8px);left:0;min-width:260px;max-width:320px;background:#1e293b;border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:8px 10px;font-size:11px;color:#b0c4d8;line-height:1.5;z-index:100;opacity:0;pointer-events:none;transform:translateY(4px);transition:opacity .2s,transform .2s;box-shadow:0 4px 16px rgba(0,0,0,.4)}

    /* --- Responsive --- */
    @media(max-width:960px){.pipeline{flex-wrap:wrap;gap:8px}.pipe-col{min-width:140px;max-width:none;flex:1 1 45%}.pipe-arrow{display:none}.detail-grid{grid-template-columns:1fr}.ml-grid{grid-template-columns:1fr 1fr}}

    /* --- Tutorial Overlay --- */
    .tut-backdrop{position:fixed;inset:0;z-index:10000;pointer-events:none;transition:opacity .3s}
    .tut-backdrop.hidden{opacity:0;display:none}
    .tut-backdrop-fill{position:absolute;inset:0;background:rgba(0,0,0,.62);pointer-events:auto}
    .tut-spotlight{position:absolute;border-radius:10px;box-shadow:0 0 0 9999px rgba(0,0,0,.62);pointer-events:none;transition:top .35s,left .35s,width .35s,height .35s;z-index:10001}
    .tut-tooltip{position:absolute;z-index:10002;background:#151d2b;color:#eaeaea;border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:16px 18px;max-width:340px;min-width:220px;box-shadow:0 8px 32px rgba(0,0,0,.5);pointer-events:auto;transition:top .35s,left .35s}
    .tut-tooltip h3{margin:0 0 6px;font-size:14px;font-weight:700}
    .tut-tooltip p{margin:0 0 12px;font-size:12.5px;color:#b0c4d8;line-height:1.5}
    .tut-tooltip .tut-step-label{font-size:10px;color:#6b7a8d;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}
    .tut-nav{display:flex;align-items:center;gap:8px}
    .tut-nav button{font-size:12px;padding:6px 14px;border-radius:8px;border:1px solid rgba(255,255,255,.15);background:transparent;color:#eaeaea;cursor:pointer}
    .tut-nav button:hover{background:rgba(255,255,255,.07)}
    .tut-nav button.tut-primary{background:#3b82f6;border-color:#3b82f6;color:#fff;font-weight:700}
    .tut-nav button.tut-primary:hover{background:#2563eb}
    .tut-nav .tut-skip{margin-left:auto;font-size:11px;color:#6b7a8d;cursor:pointer;border:none;background:none;padding:4px}
    .tut-nav .tut-skip:hover{color:#eaeaea}
    .tut-dots{display:flex;gap:4px;margin-right:8px}
    .tut-dot{width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,.15)}
    .tut-dot.active{background:#3b82f6}

    /* --- Welcome Splash --- */
    .tut-splash{position:fixed;inset:0;z-index:10010;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.72)}
    .tut-splash.hidden{display:none}
    .tut-splash-card{background:#111823;border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:32px 36px;max-width:440px;text-align:center;box-shadow:0 12px 48px rgba(0,0,0,.6)}
    .tut-splash-card h2{font-size:20px;margin:0 0 8px;font-weight:800}
    .tut-splash-card p{color:#9bb0c5;font-size:13px;line-height:1.6;margin:0 0 24px}
    .tut-splash-card .tut-begin{padding:10px 28px;font-size:14px;font-weight:700;border-radius:10px;background:#3b82f6;color:#fff;border:none;cursor:pointer}
    .tut-splash-card .tut-begin:hover{background:#2563eb}
    .tut-splash-card .tut-skip-link{display:block;margin-top:14px;color:#6b7a8d;font-size:12px;cursor:pointer;border:none;background:none}
    .tut-splash-card .tut-skip-link:hover{color:#eaeaea}

    /* --- Help button (re-launch) --- */
    .tut-help-btn{position:fixed;bottom:18px;right:18px;z-index:9998;width:36px;height:36px;border-radius:50%;background:#1e293b;border:1px solid rgba(255,255,255,.12);color:#60a5fa;font-size:16px;font-weight:800;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(0,0,0,.3)}
    .tut-help-btn:hover{background:#263548}
  </style>
</head>
<body>

<!-- ====== Header ====== -->
<header>
  <b>Bladnir Tech &mdash; PactGate&trade;</b>
  <span class="muted">Governed automation &bull; Adaptive ML &bull; Audit-first</span>
  <a href="/ame/dashboard" style="margin-left:auto;color:#60a5fa;text-decoration:none;font-size:12px">AME Trust Dashboard</a>
  <span class="muted" id="status">Loading&hellip;</span>
</header>

<!-- ====== Controls bar ====== -->
<div class="controls">
  <select id="scenarioSelect" style="min-width:180px"></select>
  <button onclick="seedScenario()">Seed</button>
  <button onclick="seedAll()">Seed All</button>
  <div style="width:1px;height:20px;background:var(--line)"></div>
  <input id="search" placeholder="Search cases&hellip;" oninput="renderBoard()" style="flex:1;min-width:100px;max-width:260px"/>
  <div style="flex:1"></div>
  <button onclick="refreshAll()">Refresh</button>
  <button onclick="resetDashboard()" style="color:#f87171;border-color:rgba(248,113,113,.3)">Reset</button>
</div>

<!-- ====== ML Intelligence Strip ====== -->
<div class="ml-strip">
  <div class="ml-strip-header">
    <span class="ml-strip-title">ML Intelligence</span>
    <button onclick="seedMlData()" style="background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.3);color:#60a5fa">Seed ML Data</button>
    <button id="trainBtn" onclick="trainModels()">Train Models</button>
    <span class="muted" id="trainStatus"></span>
    <div style="flex:1"></div>
    <span class="muted" id="mlEventCount">0 events</span>
  </div>
  <div class="ml-grid">
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotDecision"></span>Decision Predictor</div>
      <div class="ml-card-value" id="mlDecVal">Untrained</div>
      <div class="ml-card-sub" id="mlDecSub">Predicts: will human approve?</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotOutcome"></span>Outcome Predictor</div>
      <div class="ml-card-value" id="mlOutVal">Untrained</div>
      <div class="ml-card-sub" id="mlOutSub">Predicts: will action succeed?</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotAnomaly"></span>Anomaly Detector</div>
      <div class="ml-card-value" id="mlAnoVal">Untrained</div>
      <div class="ml-card-sub" id="mlAnoSub">Detects unusual event patterns</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label">Learning &amp; ROI</div>
      <div class="roi-grid">
        <div class="roi-stat"><span class="roi-num" id="roiEvents">0</span><span class="roi-label">Events</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiAuto">0</span><span class="roi-label">Auto-approved</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiHuman">0</span><span class="roi-label">Human</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiTime">0m</span><span class="roi-label">Time Saved</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ====== Anomaly Alert Banner ====== -->
<div class="anomaly-banner" id="anomalyBanner">
  <b>Anomaly Detected</b> <span id="anomalyDetail"></span>
</div>

<!-- ====== Scenario Description Bar ====== -->
<div class="scenario-desc" id="scenarioDesc"></div>

<!-- ====== Live Activity Feed ====== -->
<div class="feed-toggle" id="feedToggle" onclick="toggleFeed()">
  <span class="feed-toggle-title">Live Activity</span>
  <span class="feed-badge" id="feedCount">0</span>
  <div style="flex:1"></div>
  <span class="feed-toggle-arrow" id="feedArrow">&#9660;</span>
</div>
<div class="feed-strip" id="feedStrip">
  <div class="feed-list" id="feedList">
    <div class="feed-item" data-cat="info"><span class="feed-msg">Ready.</span> <span class="feed-detail">Seed a scenario to begin the demo. The activity feed will narrate every action as it happens.</span></div>
  </div>
</div>

<!-- ====== Pipeline board (full-width, horizontal) ====== -->
<div class="pipeline" id="board"></div>

<!-- ====== Detail panel (below pipeline, hidden until case selected) ====== -->
<div class="detail" id="detailPanel">

  <div class="detail-bar">
    <div>
      <div class="detail-title" id="caseMeta">No case selected.</div>
      <div class="detail-pills">
        <span class="pill" id="pillQueue">queue: &mdash;</span>
        <span class="pill" id="pillIns">insurance: &mdash;</span>
        <span class="pill" id="pillState">state: &mdash;</span>
        <span class="pill" id="pillAme">AME: &mdash;</span>
      </div>
    </div>
    <div class="detail-actions">
      <button onclick="autoStep()">Auto-step</button>
      <button onclick="openAuthModal()">Proposals</button>
      <button onclick="toggleJson()">JSON</button>
      <button id="repBtn" onclick="toggleRepetition()">Repeat: OFF</button>
      <label style="display:flex;align-items:center;gap:4px;font-size:11px;color:var(--muted)">
        <input type="checkbox" id="repeatTasks"> +tasks
      </label>
    </div>
  </div>

  <div class="auth-row">
    <span style="font-size:11px;font-weight:700;color:var(--muted)">Authorization Gates:</span>
    <label><input type="checkbox" id="a1" onchange="saveAuth('kroger.prescriber_approval_to_data_entry',this.checked)"> Prescriber &rarr; Data Entry</label>
    <label><input type="checkbox" id="a2" onchange="saveAuth('kroger.data_entry_to_preverify_insurance',this.checked)"> Data Entry &rarr; Pre-Verify</label>
    <label><input type="checkbox" id="a3" onchange="saveAuth('kroger.preverify_to_access_granted',this.checked)"> Pre-Verify &rarr; Cleared</label>
  </div>

  <div class="detail-grid">
    <div class="detail-card"><b>Tasks</b><div id="tasks"></div></div>
    <div class="detail-card"><b>Timeline</b><div id="events"></div></div>
  </div>

  <pre id="json" style="display:none">{}</pre>
</div>

<!-- ====== Proposal Modal ====== -->
<div id="authModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:9999">
  <div style="max-width:920px;margin:5vh auto;background:#0f1622;color:#fff;border-radius:14px;border:1px solid rgba(255,255,255,.12);padding:16px;max-height:85vh;overflow-y:auto">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
      <div>
        <div style="font-weight:800;font-size:13px">Proposal Review</div>
        <div class="muted" id="authMeta" style="margin-top:4px">&mdash;</div>
      </div>
      <div style="display:flex;gap:6px">
        <button onclick="refreshAuthModal()">Refresh</button>
        <button onclick="closeAuthModal()">Close</button>
      </div>
    </div>
    <div style="height:12px"></div>
    <div class="detail-grid">
      <div class="detail-card" style="margin:0"><b>Suggested Actions</b><div id="authSuggested"></div></div>
      <div class="detail-card" style="margin:0"><b>Proposals &amp; Audit</b><div id="authProposals"></div></div>
    </div>
  </div>
</div>

<!-- ====== Tutorial: Welcome Splash ====== -->
<div class="tut-splash hidden" id="tutSplash">
  <div class="tut-splash-card">
    <h2>Welcome to Bladnir Tech</h2>
    <p>This guided walkthrough will show you how PactGate&trade; manages pharmacy prescription workflows with governance gates, proposals, and AI-driven trust automation.</p>
    <button class="tut-begin" onclick="tutStart()">Click here to begin</button>
    <button class="tut-skip-link" onclick="tutDismiss()">Skip tutorial</button>
  </div>
</div>

<!-- ====== Tutorial: Backdrop + Spotlight + Tooltip ====== -->
<div class="tut-backdrop hidden" id="tutBackdrop">
  <div class="tut-backdrop-fill" id="tutBackdropFill"></div>
  <div class="tut-spotlight" id="tutSpotlight"></div>
  <div class="tut-tooltip" id="tutTooltip">
    <div class="tut-step-label" id="tutStepLabel">Step 1 of 10</div>
    <h3 id="tutTitle">Title</h3>
    <p id="tutBody">Body</p>
    <div class="tut-nav">
      <div class="tut-dots" id="tutDots"></div>
      <button id="tutBack" onclick="tutPrev()">Back</button>
      <button id="tutNext" class="tut-primary" onclick="tutNext()">Next</button>
      <button class="tut-skip" onclick="tutDismiss()">Skip</button>
    </div>
  </div>
</div>

<!-- ====== Tutorial: Help button (re-launch) ====== -->
<button class="tut-help-btn" id="tutHelpBtn" onclick="tutShowSplash()" title="Restart guided tutorial">?</button>

<script>
let ALL=[], AUTH={}, AME_SCOPES={}, ML_STATS=null, selected=null, repTimer=null, authOpen=false;
let SCENARIOS=[], feedOpen=false;

function setStatus(t){ document.getElementById('status').textContent=t }

/* ---------- Activity Feed ---------- */
function toggleFeed(){
  feedOpen=!feedOpen;
  document.getElementById('feedStrip').classList.toggle('open',feedOpen);
  document.getElementById('feedArrow').classList.toggle('open',feedOpen);
}
async function fetchActivityFeed(){
  try{
    const d=await api("/dashboard/api/activity?limit=40");
    const items=d.items||[];
    const el=document.getElementById('feedList');
    document.getElementById('feedCount').textContent=items.length;
    if(!items.length){el.innerHTML='<div class="feed-item" data-cat="info"><span class="feed-msg">No activity yet.</span> <span class="feed-detail">Seed a scenario and start interacting to see live narration here.</span></div>';return}
    el.innerHTML=items.map(it=>{
      const ts=it.ts?it.ts.substring(11,19):'';
      return '<div class="feed-item" data-cat="'+(it.category||'info')+'"><span class="feed-ts">'+ts+'</span><span class="feed-msg">'+esc(it.message)+'</span>'+(it.detail?'<div class="feed-detail">'+esc(it.detail)+'</div>':'')+'</div>';
    }).join('');
    /* auto-scroll to top (newest first) */
    el.scrollTop=0;
  }catch(e){console.warn("feed:",e)}
}
function esc(s){if(!s)return'';return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

async function api(path, opts={}){
  const res=await fetch(path,{headers:{"Content-Type":"application/json",...(opts.headers||{})},...opts});
  let txt="",js=null;
  try{txt=await res.text();js=txt?JSON.parse(txt):null}catch(_){}
  if(!res.ok) throw new Error((js&&(js.detail||js.message))||txt||('HTTP '+res.status));
  return js||(txt?JSON.parse(txt):{});
}

function toggleJson(){const e=document.getElementById("json");e.style.display=e.style.display==="none"?"block":"none"}

/* ---------- Scenarios ---------- */
async function loadScenarios(){
  const d=await api("/dashboard/api/scenarios");
  SCENARIOS=d.scenarios||[];
  const sel=document.getElementById("scenarioSelect");
  sel.innerHTML="";
  SCENARIOS.forEach(s=>{const o=document.createElement("option");o.value=s.id;o.textContent=s.label;sel.appendChild(o)});
  sel.onchange=showScenarioDesc;
  showScenarioDesc();
}
function showScenarioDesc(){
  const sid=document.getElementById("scenarioSelect").value;
  const s=SCENARIOS.find(x=>x.id===sid);
  const el=document.getElementById("scenarioDesc");
  if(s&&s.description){el.textContent=s.description;el.classList.add("show")}
  else{el.classList.remove("show")}
}
async function seedScenario(){
  setStatus("Seeding\u2026");
  await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:document.getElementById("scenarioSelect").value||"happy_path",seed_all:false})});
  await refreshAll();if(!feedOpen)toggleFeed();setStatus("Ready");
}
async function seedAll(){
  setStatus("Seeding all\u2026");
  await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:"happy_path",seed_all:true})});
  await refreshAll();if(!feedOpen)toggleFeed();setStatus("Ready");
}

/* ---------- AME scopes (for pipeline badges) ---------- */
async function fetchAmeScopes(){
  try{
    const scopes=await api("/ame/scopes?tenant_id=default");
    AME_SCOPES={};
    (scopes||[]).forEach(s=>{
      if(s.site_id==="dashboard_demo"&&s.action_type==="advance_queue")
        AME_SCOPES[s.queue]={stage:s.stage,trust:parseFloat(s.trust_score||0)};
    });
  }catch(e){console.warn("AME scopes:",e)}
}
function stageName(s){return{observe:"Observe",propose:"Propose",guarded_auto:"Guarded",conditional_auto:"Conditional",full_auto:"Full Auto"}[s]||s||"\u2014"}
function stageExplain(s){return{
  observe:"OBSERVE: AI watches and learns from human decisions. All actions require manual approval through governance gates.",
  propose:"PROPOSE: AI now suggests actions with confidence scores. Humans still make the final decision, but AI is learning what gets approved.",
  guarded_auto:"GUARDED AUTO: AI executes automatically with a 15-minute rollback window. Humans can undo any action within that window.",
  conditional_auto:"CONDITIONAL AUTO: AI executes when confidence and safety thresholds are met. Near-full automation with safety checks.",
  full_auto:"FULL AUTO: AI acts autonomously. Earned through consistent reliability, alignment with human decisions, and safety calibration.",
}[s]||"No trust data yet. Seed scenarios and process cases to build trust."}
function trustColor(t){if(t>=.8)return"#4ade80";if(t>=.6)return"#fbbf24";if(t>=.4)return"#60a5fa";return"#6b7280"}

/* ---------- Board ---------- */
function groupByQueue(rows){
  const c={contact_manager:[],inbound_comms:[],data_entry:[],pre_verification:[],dispensing:[],verification:[]};
  rows.forEach(r=>{const q=r.queue||"unknown";if(c[q])c[q].push(r)});return c;
}
function matchesSearch(r,s){if(!s)return true;s=s.toLowerCase();return(r.name||"").toLowerCase().includes(s)||(r.queue||"").toLowerCase().includes(s)}

const QUEUE_ORDER=[
  ["contact_manager","Contact Manager"],
  ["inbound_comms","Inbound Comms"],
  ["data_entry","Data Entry"],
  ["pre_verification","Pre-Verification"],
  ["dispensing","Dispensing"],
  ["verification","Verification"],
];

function renderBoard(){
  const s=document.getElementById("search").value.trim();
  const rows=ALL.filter(r=>matchesSearch(r,s));
  const cols=groupByQueue(rows);
  const board=document.getElementById("board");
  board.innerHTML="";

  QUEUE_ORDER.forEach(([key,title],i)=>{
    /* arrow connector */
    if(i>0){
      const ar=document.createElement("div");ar.className="pipe-arrow";
      ar.innerHTML='<svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14m0 0l-5-5m5 5l-5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      board.appendChild(ar);
    }

    const ame=AME_SCOPES[key]||{};
    const stage=ame.stage||"-";
    const trust=ame.trust||0;
    const cnt=cols[key].length;

    const col=document.createElement("div");col.className="pipe-col";

    /* header with stage tooltip */
    const hdr=document.createElement("div");hdr.className="pipe-header";
    hdr.innerHTML=
      '<div class="pipe-title">'+title+' <span style="color:var(--muted);font-weight:400">('+cnt+')</span></div>'+
      '<div class="stage-tip"><div class="pipe-badge" data-stage="'+stage+'">'+stageName(stage)+(trust>0?' \u2022 '+Math.round(trust*100)+'%':'')+'</div>'+
      '<div class="stage-tip-text">'+stageExplain(stage)+'</div></div>'+
      '<div class="pipe-trust"><div class="pipe-trust-fill" style="width:'+Math.round(trust*100)+'%;background:'+trustColor(trust)+'"></div></div>';
    col.appendChild(hdr);

    /* body: case cards with scenario labels */
    const body=document.createElement("div");body.className="pipe-body";
    cols[key].forEach(r=>{
      const card=document.createElement("div");
      card.className="pipe-card"+(selected&&selected.id===r.id?" active":"");
      card.onclick=()=>selectCase(r.id);
      const short=(r.name||"").replace("Kroger \u2022 ","").replace(" (Demo)","");
      let scenarioTag='';
      if(r.scenario_label){scenarioTag='<div class="pipe-card-scenario">'+esc(r.scenario_label.split(' — ')[0]||r.scenario_label)+'</div>';}
      card.innerHTML='<div class="pipe-card-name">'+short+'</div>'+scenarioTag+'<div class="pipe-card-ins">'+(r.insurance||"\u2014")+'</div>';
      body.appendChild(card);
    });
    col.appendChild(body);
    board.appendChild(col);
  });
}

/* ---------- Detail panel ---------- */
function renderDetails(wf){
  selected=wf;
  document.getElementById("detailPanel").classList.add("open");
  document.getElementById("caseMeta").textContent="#"+wf.id+" \u2022 "+wf.name;
  document.getElementById("pillQueue").textContent="queue: "+wf.queue;
  document.getElementById("pillIns").textContent="insurance: "+wf.insurance;
  document.getElementById("pillState").textContent="state: "+wf.state;

  const ame=AME_SCOPES[wf.queue]||{};
  document.getElementById("pillAme").textContent="AME: "+stageName(ame.stage)+(ame.trust>0?" ("+Math.round(ame.trust*100)+"%)":"");

  document.getElementById("json").textContent=JSON.stringify(wf.raw,null,2);

  /* tasks */
  const tEl=document.getElementById("tasks");tEl.innerHTML="";
  (wf.raw.tasks||[]).forEach(t=>{const d=document.createElement("div");d.className="item";d.innerHTML="<b>"+t.name+"</b><div class='muted'>assigned: "+(t.assigned_to||"\u2014")+" \u2022 state: "+t.state+"</div>";tEl.appendChild(d)});
  if(!(wf.raw.tasks||[]).length) tEl.innerHTML='<div class="muted">No tasks</div>';

  /* events */
  const eEl=document.getElementById("events");eEl.innerHTML="";
  (wf.raw.events||[]).slice().reverse().forEach(e=>{const d=document.createElement("div");d.className="item";d.innerHTML="<b>"+e.event_type+"</b><div class='muted'>"+JSON.stringify(e.payload||{})+"</div>";eEl.appendChild(d)});
  if(!(wf.raw.events||[]).length) eEl.innerHTML='<div class="muted">No events</div>';

  /* auth checkboxes */
  document.getElementById("a1").checked=!!AUTH["kroger.prescriber_approval_to_data_entry"];
  document.getElementById("a2").checked=!!AUTH["kroger.data_entry_to_preverify_insurance"];
  document.getElementById("a3").checked=!!AUTH["kroger.preverify_to_access_granted"];

  renderBoard(); /* refresh active highlight */
}

async function selectCase(id){const wf=ALL.find(x=>x.id===id);if(wf)renderDetails(wf)}

/* ---------- Auth ---------- */
async function saveAuth(key,enabled){
  setStatus("Saving\u2026");
  const res=await api("/dashboard/api/automation",{method:"POST",body:JSON.stringify({transition_key:key,enabled})});
  AUTH=res.authorizations||{};setStatus("Ready");
}

/* ---------- Auto-step ---------- */
async function autoStep(){
  if(!selected)return alert("Select a case first.");
  setStatus("Auto-stepping\u2026");
  try{
    await api("/dashboard/api/auto-step",{method:"POST",body:JSON.stringify({workflow_id:selected.id})});
    await refreshAll();setStatus("Ready");
  }catch(e){
    const m=String(e?.message||"");
    if(m.toLowerCase().includes("not authorized")||m.includes("403")){setStatus("Requires authorization");return}
    setStatus("Error");alert("Auto-step failed: "+m);
  }
}

/* ---------- Repetition ---------- */
async function simulateOnce(){
  if(!selected)return;
  await api("/dashboard/api/simulate",{method:"POST",body:JSON.stringify({workflow_id:selected.id,cycles:1,repeat_tasks:document.getElementById("repeatTasks").checked})});
  await refreshAll();
}
function toggleRepetition(){
  const btn=document.getElementById("repBtn");
  if(repTimer){clearInterval(repTimer);repTimer=null;btn.textContent="Repeat: OFF";setStatus("Ready");return}
  if(!selected){alert("Select a case first.");return}
  btn.textContent="Repeat: ON";setStatus("Repeating\u2026");
  repTimer=setInterval(()=>{simulateOnce().catch(()=>{clearInterval(repTimer);repTimer=null;btn.textContent="Repeat: OFF";setStatus("Ready")})},3000);
}

/* ---------- ML Intelligence ---------- */
async function fetchMlStats(){
  try{
    ML_STATS=await api("/dashboard/api/ml-stats");
    renderMlStrip();
  }catch(e){console.warn("ML stats:",e)}
}

function renderMlStrip(){
  if(!ML_STATS)return;
  const m=ML_STATS.models||{};
  const ev=ML_STATS.events||{};
  const roi=ML_STATS.roi||{};

  /* Decision */
  const dec=m.decision||{};
  const dotDec=document.getElementById("dotDecision");
  const decVal=document.getElementById("mlDecVal");
  const decSub=document.getElementById("mlDecSub");
  if(dec.trained){
    dotDec.className="ml-dot on";
    const acc=((dec.metrics||{}).accuracy||0)*100;
    decVal.textContent="v"+dec.version+" \\u00b7 "+acc.toFixed(1)+"%";
    decSub.textContent="Accuracy \\u00b7 trained on "+dec.training_events+" decisions";
  }else{
    dotDec.className="ml-dot off";
    decVal.textContent="Untrained";
    decSub.textContent="Need 50+ decided proposals";
  }

  /* Outcome */
  const out=m.outcome||{};
  const dotOut=document.getElementById("dotOutcome");
  const outVal=document.getElementById("mlOutVal");
  const outSub=document.getElementById("mlOutSub");
  if(out.trained){
    dotOut.className="ml-dot on";
    const acc=((out.metrics||{}).accuracy||0)*100;
    outVal.textContent="v"+out.version+" \\u00b7 "+acc.toFixed(1)+"%";
    outSub.textContent="Accuracy \\u00b7 replaces hardcoded values";
  }else{
    dotOut.className="ml-dot off";
    outVal.textContent="Untrained";
    outSub.textContent="Need 30+ observed outcomes";
  }

  /* Anomaly */
  const ano=m.anomaly||{};
  const dotAno=document.getElementById("dotAnomaly");
  const anoVal=document.getElementById("mlAnoVal");
  const anoSub=document.getElementById("mlAnoSub");
  if(ano.trained){
    dotAno.className="ml-dot on";
    const pct=(ano.metrics||{}).anomaly_percentage||0;
    anoVal.textContent="v"+ano.version+" \\u00b7 "+pct+"%";
    anoSub.textContent="Baseline anomaly rate \\u00b7 "+(ano.training_windows||0)+" windows";
  }else{
    dotAno.className="ml-dot off";
    anoVal.textContent="Untrained";
    anoSub.textContent="Need sufficient event history";
  }

  /* ROI */
  document.getElementById("roiEvents").textContent=ev.total||0;
  document.getElementById("roiAuto").textContent=roi.auto_approved||0;
  document.getElementById("roiHuman").textContent=roi.human_reviewed||0;
  document.getElementById("roiTime").textContent=(roi.time_saved_min||0)+"m";
  document.getElementById("mlEventCount").textContent=(ev.total||0)+" events logged";
}

async function trainModels(){
  const btn=document.getElementById("trainBtn");
  const st=document.getElementById("trainStatus");
  btn.disabled=true;btn.textContent="Training\\u2026";st.textContent="";
  try{
    const r=await api("/ame/ml/retrain",{method:"POST",body:"{}"});
    const res=r.results||{};
    let parts=[];
    if(res.decision&&res.decision.trained)parts.push("Decision: "+(res.decision.accuracy*100).toFixed(0)+"%");
    if(res.outcome&&res.outcome.trained)parts.push("Outcome: "+(res.outcome.accuracy*100).toFixed(0)+"%");
    if(res.anomaly&&res.anomaly.trained)parts.push("Anomaly: trained");
    st.textContent=parts.length?parts.join(" \\u00b7 "):"Not enough data yet";
    await fetchMlStats();
  }catch(e){
    st.textContent="Training failed: "+e.message;
  }finally{
    btn.disabled=false;btn.textContent="Train Models";
  }
}

async function seedMlData(){
  const st=document.getElementById("trainStatus");
  st.textContent="Seeding training data + training models\\u2026";
  try{
    const r=await api("/dashboard/api/seed-ml",{method:"POST",body:"{}"});
    const res=r.training_results||{};
    let parts=[];
    if(res.decision&&res.decision.trained)parts.push("Decision: "+((res.decision.metrics||{}).accuracy*100).toFixed(0)+"%");
    if(res.outcome&&res.outcome.trained)parts.push("Outcome: "+((res.outcome.metrics||{}).accuracy*100).toFixed(0)+"%");
    if(res.anomaly&&res.anomaly.trained)parts.push("Anomaly: trained");
    st.textContent=r.events_created+" events seeded"+(parts.length?" \\u00b7 "+parts.join(" \\u00b7 "):"");
    await fetchMlStats();
  }catch(e){st.textContent="Seed failed: "+e.message}
}

function renderMlPrediction(mlDec,mlOut){
  if(!mlDec&&!mlOut)return"";
  let h='<div class="ml-pred">';
  if(mlDec&&mlDec.approve_probability!==undefined&&mlDec.confidence_band!=="cold_start"){
    const pct=Math.round(mlDec.approve_probability*100);
    const c=pct>=85?"#4ade80":pct>=50?"#fbbf24":"#f87171";
    h+='<div class="ml-pred-row"><span>Approve: <b style="color:'+c+'">'+pct+'%</b> ('+mlDec.confidence_band+')</span>';
    h+='<div class="ml-pred-bar"><div class="ml-pred-fill" style="width:'+pct+'%;background:'+c+'"></div></div></div>';
    if(mlDec.top_factors&&mlDec.top_factors.length){
      h+='<div class="ml-factors">Factors: '+mlDec.top_factors.map(f=>f[0]+"("+(f[1]>0?"+":"")+f[1]+")").join(", ")+"</div>";
    }
  }
  if(mlOut&&mlOut.success_probability!==undefined&&!mlOut.cold_start){
    const pct=Math.round(mlOut.success_probability*100);
    const s=Math.round((mlOut.safety_score||0)*100);
    h+='<div class="ml-pred-row" style="margin-top:4px"><span>Success: <b>'+pct+'%</b> \\u00b7 Safety: <b>'+s+'%</b></span></div>';
  }
  h+="</div>";
  return h;
}

/* ---------- Refresh ---------- */
async function refreshAll(){
  setStatus("Loading\u2026");
  try{
    const [d1,d2]=await Promise.all([
      api("/dashboard/api/workflows").catch(e=>{console.error("workflows:",e);return{workflows:[]}}),
      api("/dashboard/api/automation").catch(e=>{console.error("automation:",e);return{authorizations:{}}})
    ]);
    ALL=d1.workflows||[];AUTH=d2.authorizations||{};
    await Promise.all([fetchAmeScopes(),fetchMlStats().catch(()=>{}),fetchActivityFeed().catch(()=>{})]);
    renderBoard();
    if(selected){const wf=ALL.find(x=>x.id===selected.id);if(wf)renderDetails(wf)}
    if(authOpen)refreshAuthModal().catch(()=>{});
    setStatus("Ready");
  }catch(e){console.error("refreshAll:",e);setStatus("Error — check console")}
}

/* ---------- Proposal modal ---------- */
function closeAuthModal(){authOpen=false;document.getElementById("authModal").style.display="none"}
async function openAuthModal(){
  if(!selected)return alert("Select a case first.");
  authOpen=true;document.getElementById("authModal").style.display="block";await refreshAuthModal();
}
async function refreshAuthModal(){
  if(!selected)return;
  const d=await api("/dashboard/api/cases/"+selected.id);
  const c=d.case||{};
  document.getElementById("authMeta").textContent="Case #"+c.id+" \u2022 "+(c.name||"")+" \u2022 queue: "+c.queue;

  /* suggested */
  const sug=d.suggested_actions||[];const sugEl=document.getElementById("authSuggested");
  if(!sug.length){sugEl.innerHTML='<div class="muted">No suggestions.</div>'}
  else{
    sugEl.innerHTML=sug.map(a=>{
      const enc=encodeURIComponent(JSON.stringify(a));
      const mlHtml=renderMlPrediction(null,a.ml_outcome||null);
      return '<div class="item"><b>'+a.label+'</b><div class="muted" style="margin-top:4px">confidence: '+(a.confidence||0).toFixed(2)+' \u2022 safety: '+(a.safety_score||0).toFixed(2)+'</div>'+mlHtml+'<div style="height:8px"></div><button class="js-cp" data-enc="'+enc+'">Create Proposal</button></div>';
    }).join("");
    sugEl.querySelectorAll(".js-cp").forEach(b=>{b.onclick=()=>{createProposal(JSON.parse(decodeURIComponent(b.getAttribute("data-enc"))))}});
  }

  /* proposals */
  const props=d.proposals||[];const pEl=document.getElementById("authProposals");
  if(!props.length){pEl.innerHTML='<div class="muted">No proposals yet.</div>'}
  else{
    pEl.innerHTML=props.map(p=>{
      const a=p.action||{},st=p.status||"unknown";
      const aud=(p.audit||[]).slice().reverse().slice(0,4).map(x=>'<div class="muted">\u2022 '+x.ts+': '+x.event+'</div>').join("");
      const mlHtml=renderMlPrediction(p.ml_decision||null,p.ml_outcome||null);
      let btns="";
      if(st==="pending")btns='<button onclick="decideProposal(\\''+p.id+'\\',\\'approve\\')">Approve</button> <button onclick="decideProposal(\\''+p.id+'\\',\\'reject\\')">Reject</button>';
      else if(st==="approved")btns='<button class="primary" onclick="executeProposal(\\''+p.id+'\\')">Execute</button>';
      else btns='<span class="pill">'+st+'</span>';
      return '<div class="item"><b>'+p.id+'</b> <span class="pill" style="margin-left:6px">'+st+'</span><div class="muted" style="margin-top:4px">'+(a.label||a.type||"action")+'</div>'+mlHtml+'<div style="height:6px"></div><div style="display:flex;gap:6px">'+btns+'</div><div style="height:6px"></div>'+aud+'</div>';
    }).join("");
  }
}
async function createProposal(a){
  if(!selected)return;
  await api("/dashboard/api/cases/"+selected.id+"/propose",{method:"POST",body:JSON.stringify({action_type:a.type,label:a.label,payload:a.payload||{},confidence:a.confidence??0.75,safety_score:a.safety_score??0.75})});
  await refreshAll();await refreshAuthModal();
}
async function decideProposal(pid,decision){
  await api("/dashboard/api/automation/"+pid+"/decide",{method:"POST",body:JSON.stringify({decision,decided_by:"Pharmacy_Manager",note:""})});
  await refreshAll();await refreshAuthModal();
}
async function executeProposal(pid){
  try{await api("/dashboard/api/automation/"+pid+"/execute",{method:"POST",body:JSON.stringify({executed_by:"system"})})}catch(e){alert(e.message)}
  await refreshAll();await refreshAuthModal();
}

/* ---------- Reset ---------- */
async function resetDashboard(){
  if(!confirm("Reset everything? This clears all cases, proposals, AME trust data, and governance gates."))return;
  setStatus("Resetting\u2026");
  try{
    await api("/dashboard/api/reset",{method:"POST",body:"{}"});
    selected=null;
    document.getElementById("detailPanel").classList.remove("open");
    await refreshAll();
    setStatus("Reset complete");
  }catch(e){console.error("reset:",e);setStatus("Reset failed")}
}

/* ================================================================
   GUIDED TUTORIAL ENGINE
   ================================================================ */

const TUT_STEPS = [
  {
    title: "Seed Demo Data",
    body: "First, let's populate the board. Click <b>Seed All</b> to create 5 different pharmacy scenarios — each starts at a different queue so you can see the full pipeline in action.",
    target: ()=>document.querySelector('.controls button:nth-child(3)'),
    position: "bottom",
    waitForClick: true,
    onArrive: null,
  },
  {
    title: "The Pipeline Board",
    body: "Cases flow left-to-right through pharmacy queues. Each column shows its <b>AME trust stage</b> (hover the badge for details) and trust score. Notice how different scenarios land at different stages of the workflow.",
    target: ()=>document.getElementById('board'),
    position: "bottom",
  },
  {
    title: "Select a Case",
    body: "Click any case card in the pipeline to view its details. Try clicking one of the cases in the <b>Inbound Comms</b> column.",
    target: ()=>document.querySelector('.pipe-card'),
    position: "bottom",
    waitForClick: true,
  },
  {
    title: "Case Detail Panel",
    body: "Here you can see the case name, current queue, insurance status, tasks, and timeline. The <b>Authorization Gates</b> row controls which queue transitions are permitted.",
    target: ()=>document.getElementById('detailPanel'),
    position: "top",
  },
  {
    title: "Enable a Governance Gate",
    body: "Governance gates require explicit human authorization before a case can advance. Check the <b>Prescriber -> Data Entry</b> checkbox to authorize this transition.",
    target: ()=>document.querySelector('.auth-row'),
    position: "top",
    waitForClick: true,
  },
  {
    title: "Open Proposals",
    body: "Now click <b>Proposals</b> to see suggested automation actions for this case. The system can propose advancing the case to the next queue.",
    target: ()=>{const btns=document.querySelectorAll('.detail-actions button');return btns[1]||null},
    position: "bottom",
    waitForClick: true,
  },
  {
    title: "Create a Proposal",
    body: "In the Suggested Actions panel, click <b>Create Proposal</b> to have the system formally propose advancing this case. This enters the governance approval flow.",
    target: ()=>document.querySelector('#authSuggested .js-cp')||document.getElementById('authSuggested'),
    position: "left",
    waitForClick: true,
  },
  {
    title: "Approve the Proposal",
    body: "Now click <b>Approve</b> on the pending proposal in the right panel. In production, this would be a pharmacist or manager making the decision.",
    target: ()=>document.getElementById('authProposals'),
    position: "left",
    waitForClick: true,
  },
  {
    title: "Execute the Proposal",
    body: "The proposal is now approved. Click <b>Execute</b> to carry out the action. Watch the case move to the next queue in the pipeline!",
    target: ()=>document.getElementById('authProposals'),
    position: "left",
    waitForClick: true,
  },
  {
    title: "Try Auto-Step",
    body: "Close the proposal modal and click <b>Auto-step</b> to let the AME trust system automatically advance the case. This uses ML predictions and governance gates.",
    target: ()=>document.querySelector('.detail-actions button'),
    position: "bottom",
  },
  {
    title: "AME Trust Dashboard",
    body: "Click the <b>AME Trust Dashboard</b> link in the header to see how trust scores evolve over time as the system learns from human decisions.",
    target: ()=>document.querySelector('header a'),
    position: "bottom",
  },
  {
    title: "Live Activity Feed",
    body: "The <b>Live Activity</b> bar narrates every action in plain English — proposals, decisions, executions, trust changes, and governance events. Click it to expand and follow along as you interact.",
    target: ()=>document.getElementById('feedToggle'),
    position: "bottom",
  },
  {
    title: "You're All Set!",
    body: "You now know the core workflow: <b>Seed \u2192 Select \u2192 Gate \u2192 Propose \u2192 Approve \u2192 Execute</b>. Watch the Live Activity feed and AME trust levels as the system learns. Try different scenarios, toggle gates, and seed ML data to see predictions appear. Click <b>?</b> anytime to restart this tour.",
    target: null,
    position: "center",
  },
];

let tutStep = 0;
let tutActive = false;

function tutShowSplash(){
  document.getElementById('tutSplash').classList.remove('hidden');
}
function tutHideSplash(){
  document.getElementById('tutSplash').classList.add('hidden');
}

function tutStart(){
  tutHideSplash();
  tutStep = 0;
  tutActive = true;
  document.getElementById('tutBackdrop').classList.remove('hidden');
  tutRender();
}

function tutDismiss(){
  tutActive = false;
  tutHideSplash();
  document.getElementById('tutBackdrop').classList.add('hidden');
  localStorage.setItem('bladnir_tut_done','1');
}

function tutNext(){
  if(tutStep < TUT_STEPS.length - 1){ tutStep++; tutRender(); }
  else { tutDismiss(); }
}
function tutPrev(){
  if(tutStep > 0){ tutStep--; tutRender(); }
}

function tutRender(){
  if(!tutActive) return;
  const step = TUT_STEPS[tutStep];
  const total = TUT_STEPS.length;

  /* step label */
  document.getElementById('tutStepLabel').textContent = 'Step ' + (tutStep+1) + ' of ' + total;
  document.getElementById('tutTitle').textContent = step.title;
  document.getElementById('tutBody').innerHTML = step.body;

  /* dots */
  const dotsEl = document.getElementById('tutDots');
  dotsEl.innerHTML = '';
  for(let i=0;i<total;i++){
    const d = document.createElement('div');
    d.className = 'tut-dot' + (i===tutStep?' active':'');
    dotsEl.appendChild(d);
  }

  /* back/next labels */
  document.getElementById('tutBack').style.display = tutStep===0?'none':'inline-block';
  const nextBtn = document.getElementById('tutNext');
  if(tutStep===total-1){ nextBtn.textContent='Finish'; }
  else if(step.waitForClick){ nextBtn.textContent='Next'; }
  else{ nextBtn.textContent='Next'; }

  /* target element */
  const targetEl = step.target ? step.target() : null;
  const spotlight = document.getElementById('tutSpotlight');
  const tooltip = document.getElementById('tutTooltip');
  const backdropFill = document.getElementById('tutBackdropFill');

  if(targetEl){
    /* position spotlight over target */
    const rect = targetEl.getBoundingClientRect();
    const pad = 8;
    spotlight.style.display = 'block';
    spotlight.style.top = (rect.top - pad + window.scrollY) + 'px';
    spotlight.style.left = (rect.left - pad) + 'px';
    spotlight.style.width = (rect.width + pad*2) + 'px';
    spotlight.style.height = (rect.height + pad*2) + 'px';
    backdropFill.style.display = 'none'; /* spotlight provides the backdrop via box-shadow */

    /* position tooltip */
    const pos = step.position || 'bottom';
    const tw = 340; /* tooltip max-width */

    if(pos === 'bottom'){
      tooltip.style.top = (rect.bottom + pad + 12 + window.scrollY) + 'px';
      tooltip.style.left = Math.max(12, Math.min(rect.left, window.innerWidth - tw - 20)) + 'px';
    } else if(pos === 'top'){
      tooltip.style.top = Math.max(12, (rect.top - pad - 12 + window.scrollY - 180)) + 'px';
      tooltip.style.left = Math.max(12, Math.min(rect.left, window.innerWidth - tw - 20)) + 'px';
    } else if(pos === 'left'){
      tooltip.style.top = (rect.top + window.scrollY) + 'px';
      tooltip.style.left = Math.max(12, rect.left - tw - 20) + 'px';
    } else if(pos === 'right'){
      tooltip.style.top = (rect.top + window.scrollY) + 'px';
      tooltip.style.left = (rect.right + 16) + 'px';
    }

    /* scroll target into view if needed */
    targetEl.scrollIntoView({behavior:'smooth', block:'nearest', inline:'nearest'});
  } else {
    /* center tooltip (no target) */
    spotlight.style.display = 'none';
    backdropFill.style.display = 'block';
    tooltip.style.top = '50%';
    tooltip.style.left = '50%';
    tooltip.style.transform = 'translate(-50%,-50%)';
    setTimeout(()=>{ tooltip.style.transform=''; }, 0); /* reset after positioning */
    tooltip.style.top = Math.max(100, window.innerHeight/2 - 100 + window.scrollY) + 'px';
    tooltip.style.left = Math.max(20, window.innerWidth/2 - 170) + 'px';
  }

  if(step.onArrive) step.onArrive();
}

/* Re-render on scroll/resize so spotlight tracks the element */
window.addEventListener('scroll', ()=>{ if(tutActive) tutRender(); }, {passive:true});
window.addEventListener('resize', ()=>{ if(tutActive) tutRender(); });

/* Auto-show splash on first visit */
(function(){
  if(!localStorage.getItem('bladnir_tut_done')){
    /* Wait for data to load, then show splash */
    setTimeout(tutShowSplash, 800);
  }
})();

/* ---------- Boot ---------- */
(async()=>{
  try{await loadScenarios()}catch(e){console.error("loadScenarios:",e)}
  try{await refreshAll()}catch(e){console.error("refreshAll:",e);setStatus("Error — check console")}
})();
</script>
</body>
</html>
    """
