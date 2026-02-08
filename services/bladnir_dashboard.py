


from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
from typing import Dict, Any, List
import os
import threading
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
_feed_lock = threading.Lock()

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
    with _feed_lock:
        ACTIVITY_FEED.append(entry)
        if len(ACTIVITY_FEED) > _FEED_MAX:
            ACTIVITY_FEED.pop(0)

def _stage_label(stage: str) -> str:
    return {
        "observe": "Level 1 — Learning: AI watches your team work and learns patterns",
        "propose": "Level 2 — Suggesting: AI recommends next steps, your team decides",
        "guarded_auto": "Level 3 — Assisting: AI acts automatically, humans can undo for 15 min",
        "conditional_auto": "Level 4 — Semi-Autonomous: AI handles routine cases when confidence is high",
        "full_auto": "Level 5 — Autonomous: AI handles this step end-to-end with full audit trail",
    }.get(stage, stage or "unknown")

def _short_stage(stage: str) -> str:
    return {"observe": "Learning", "propose": "Suggesting", "guarded_auto": "Assisting",
            "conditional_auto": "Semi-Auto", "full_auto": "Autonomous"}.get(stage, stage or "—")

# Industry-specific queue label mapping (Python side, for narration)
_QUEUE_LABELS: dict[str, dict[str, str]] = {
    "pharmacy": {"contact_manager": "Intake", "inbound_comms": "Triage", "data_entry": "Data Entry",
                 "pre_verification": "Verification", "dispensing": "Processing", "verification": "Complete"},
    "insurance": {"contact_manager": "Intake", "inbound_comms": "Triage", "data_entry": "Assessment",
                  "pre_verification": "Review", "dispensing": "Approval", "verification": "Settled"},
    "hr": {"contact_manager": "Request", "inbound_comms": "Documents", "data_entry": "Background Check",
           "pre_verification": "Provisioning", "dispensing": "Training", "verification": "Complete"},
}

def _q_label(queue: str, industry: str = "pharmacy") -> str:
    """Translate a raw queue key to an industry-specific display label."""
    return _QUEUE_LABELS.get(industry, _QUEUE_LABELS["pharmacy"]).get(queue, queue)

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
            f"Safety checkpoint enabled: {gate_label}",
            "gate",
            f"A manager opened this pathway. Cases can now move through this step when approved.",
        )
    else:
        governance.revoke_gate(transition_key, actor=decided_by, note=note)
        _narrate(
            f"Safety checkpoint paused: {gate_label}",
            "gate",
            f"A manager blocked this pathway. No cases will advance through this step until it's re-enabled.",
        )
    return dashboard_get_automation()

DEMO_SCENARIOS = {
    "happy_path": {
        "label": "Happy Path — Standard Refill",
        "description": "A routine prescription refill with valid insurance. The ideal case: everything checks out and the prescription flows straight through. Watch how each step is governed and approved before advancing.",
        "start_queue": "inbound_comms",
        "insurance_result": "accepted",
        "refills_ok": True,
        "has_insurance": True,
        "seed_narrative": "New prescription refill received — insurance verified and active. This is the fastest path through the pipeline. In a manual workflow, this takes 3-5 minutes of staff time. Watch how PactGate learns to handle these automatically.",
        "manual_time_sec": 240,
    },
    "insurance_rejected_outdated": {
        "label": "Insurance Rejected — Patient Outreach",
        "description": "Insurance check fails — patient's info is outdated. Staff must pause, contact the patient, and wait for updated information. This is where errors and delays cost real money. PactGate catches this early.",
        "start_queue": "data_entry",
        "insurance_result": "rejected",
        "reject_reason": "Outdated or missing insurance information",
        "refills_ok": True,
        "has_insurance": True,
        "patient_message": {
            "type": "insurance_update_request",
            "template": "Your insurance info appears outdated or missing. Please upload/confirm your active plan to continue."
        },
        "seed_narrative": "Insurance rejected — outdated information on file. In a manual process, staff might not catch this until dispensing, wasting 15+ minutes. PactGate flags it early and blocks advancement until resolved.",
        "manual_time_sec": 900,
    },
    "prior_auth_required": {
        "label": "Prior Authorization — Multi-Day Wait",
        "description": "The insurer requires pre-approval before this medication can be dispensed. Typically a 2-day wait. PactGate tracks the status and learns which medications commonly need PA, so it can warn staff up front.",
        "start_queue": "pre_verification",
        "insurance_result": "pa_required",
        "refills_ok": True,
        "has_insurance": True,
        "pa": {"eta_days": 2},
        "patient_message": {
            "type": "prior_auth_notice",
            "template": "Your plan requires prior authorization. We've initiated PA; we'll update you as soon as we hear back."
        },
        "seed_narrative": "Prior authorization required — estimated 2 business days. Manual tracking of PA status typically costs 10+ minutes per case in follow-up calls. PactGate automates the monitoring and notifies staff when resolved.",
        "manual_time_sec": 600,
    },
    "no_refills_prescriber": {
        "label": "No Refills — Prescriber Contact",
        "description": "Patient needs a refill but has zero remaining. Staff must contact the prescribing doctor and wait for a response. PactGate automates the outreach and tracks response times so you can follow up efficiently.",
        "start_queue": "contact_manager",
        "insurance_result": "accepted",
        "refills_ok": False,
        "has_insurance": True,
        "prescriber_request": {
            "type": "refill_request",
            "template": "No refills remaining. Refill request sent to prescriber."
        },
        "seed_narrative": "No refills remaining — contacting prescriber. Manual prescriber outreach averages 8 minutes per case. PactGate learns response patterns and can eventually auto-generate outreach requests.",
        "manual_time_sec": 480,
    },
    "no_insurance_discount_card": {
        "label": "Cash Pay — Discount Card Applied",
        "description": "Patient has no insurance. Staff must find the best discount program and apply it manually — a tedious, error-prone process. PactGate learns which programs match which medications and auto-applies them.",
        "start_queue": "data_entry",
        "insurance_result": "no_insurance",
        "refills_ok": True,
        "has_insurance": False,
        "discount_card": {"program": "DemoRxSaver", "bin": "999999", "pcn": "DEMO", "group": "SAVER", "member": "DEMO1234"},
        "patient_message": {
            "type": "discount_card_applied",
            "template": "No active insurance found. We applied a discount card to help complete your prescription."
        },
        "seed_narrative": "No insurance on file — discount card auto-applied. Matching discount programs manually takes 5+ minutes and often fails. As PactGate learns, this entire step becomes automatic.",
        "manual_time_sec": 360,
    },
    # --- Cross-Industry: Insurance Claims ---
    "insurance_claims": {
        "label": "Insurance Claim — Auto-Adjudication",
        "description": "An insurance claim arrives for processing. The system triages it: routine claims get fast-tracked, complex claims go to specialist review. Shows how PactGate works beyond pharmacy — any governed workflow.",
        "start_queue": "inbound_comms",
        "insurance_result": "accepted",
        "refills_ok": True,
        "has_insurance": True,
        "seed_narrative": "New insurance claim submitted for processing. In a manual workflow, an adjuster spends 12+ minutes per routine claim. PactGate learns which claims are routine and auto-adjudicates them safely.",
        "manual_time_sec": 720,
        "industry": "insurance",
    },
    # --- Cross-Industry: HR Onboarding ---
    "hr_onboarding": {
        "label": "HR Onboarding — New Employee",
        "description": "A new employee needs system access, equipment, and compliance training. Each step requires approval from different departments. PactGate orchestrates the entire flow with audit trail.",
        "start_queue": "contact_manager",
        "insurance_result": "accepted",
        "refills_ok": True,
        "has_insurance": True,
        "seed_narrative": "New employee onboarding initiated. Manual onboarding averages 4 hours across IT, HR, and facilities. PactGate coordinates approvals and learns which steps can be parallelized or automated.",
        "manual_time_sec": 14400,
        "industry": "hr",
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
    with _feed_lock:
        items = list(reversed(ACTIVITY_FEED[-limit:]))
    return {"items": items, "count": len(items)}


@router.get("/dashboard/api/report")
def generate_report(db=Depends(get_db)):
    """
    Generate a detailed report of all demo activity:
    cases, proposals, decisions, executions, and AI trust status.
    """
    from src.enterprise.ame.models import AMETrustScope

    # Gather all cases
    cases = []
    for row in DEMO_ROWS:
        industry = row.get("industry", "pharmacy")
        q_lbl = _q_label(row.get("queue", ""), industry)
        cases.append({
            "id": row["id"],
            "name": row.get("name", ""),
            "industry": industry,
            "queue": q_lbl,
            "queue_raw": row.get("queue", ""),
            "state": row.get("state", ""),
            "scenario_label": row.get("scenario_label", ""),
            "tasks": row.get("raw", {}).get("tasks", []),
            "events": row.get("raw", {}).get("events", []),
        })

    # Gather all proposals with full audit trails
    proposals = []
    for p in DEMO_PROPOSALS:
        case_row = DEMO_BY_ID.get(p.get("case_id"))
        case_name = case_row.get("name", f"Case #{p.get('case_id')}") if case_row else f"Case #{p.get('case_id')}"
        proposals.append({
            "id": p["id"],
            "case_id": p.get("case_id"),
            "case_name": case_name,
            "action": p.get("action", {}),
            "status": p.get("status", "unknown"),
            "proposed_by": p.get("proposed_by", "—"),
            "proposed_at": p.get("proposed_at", ""),
            "approved_by": p.get("approved_by"),
            "approved_at": p.get("approved_at"),
            "executed_by": p.get("executed_by"),
            "executed_at": p.get("executed_at"),
            "audit": p.get("audit", []),
        })

    # Gather AME trust scopes
    trust_scopes = []
    try:
        scopes = db.query(AMETrustScope).all()
        for s in scopes:
            trust_scopes.append({
                "queue": _q_label(s.queue or "", "pharmacy"),
                "queue_raw": s.queue or "",
                "stage": _short_stage(s.stage or "observe"),
                "trust_score": round((s.trust_score or 0) * 100),
                "proposals": s.proposal_count or 0,
                "executions": s.execution_count or 0,
                "override_rate": round((s.override_rate or 0) * 100),
            })
    except Exception:
        pass

    # Activity log (full)
    activity = list(reversed(ACTIVITY_FEED[:]))

    # Summary stats
    total_proposals = len(proposals)
    approved = sum(1 for p in proposals if p["status"] in ("approved", "executed"))
    rejected = sum(1 for p in proposals if p["status"] == "rejected")
    executed = sum(1 for p in proposals if p["status"] == "executed")
    pending = sum(1 for p in proposals if p["status"] == "pending")

    return {
        "generated_at": _now_iso(),
        "summary": {
            "total_cases": len(cases),
            "total_proposals": total_proposals,
            "approved": approved,
            "rejected": rejected,
            "executed": executed,
            "pending": pending,
        },
        "cases": cases,
        "proposals": proposals,
        "trust_scopes": trust_scopes,
        "activity_log": activity,
    }


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
        ml_note = f" Based on past decisions, {round(ml_decision['approve_probability'] * 100)}% likely to be approved."
    if ml_outcome and ml_outcome.get("success_probability") is not None and not ml_outcome.get("cold_start"):
        ml_note += f" {round(ml_outcome['success_probability'] * 100)}% chance of success."
    _narrate(
        f"AI recommends: advance {case_name}",
        "proposal",
        f"Automation level: {_short_stage(mode)}. Confidence: {round(confidence * 100)}%.{ml_note} "
        f"Waiting for your team's approval before proceeding.",
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
                f"Approved by {decided_by}: {case_name}",
                "decision",
                f"Your team confirmed this action is correct. The AI learns from this — "
                f"each approval builds confidence so similar cases can eventually be handled automatically.",
            )
        else:
            _narrate(
                f"Rejected by {decided_by}: {case_name}",
                "decision",
                f"Your team said no. The AI learns from this too — "
                f"rejections teach it to be more cautious with similar cases.",
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
            f"AI handled this automatically (automation level: {_short_stage(mode)}, "
            f"confidence: {round(meta.get('trust', 0) * 100)}%)."
        )
        if needs_rollback:
            gov_note += " Your team can undo this within 15 minutes if needed."
    else:
        gov_note = "Your team approved this step before it was executed."
    _narrate(
        f"Case advanced: {case_name} moved to {_q_label(to_q)} (+30s saved vs. manual)",
        "execution",
        f"{gov_note} Each successful step builds the AI's track record toward more autonomy.",
        meta={"ame_stage": mode, "trust": ame.get("trust", 0)},
    )

    # Check if trust stage might have changed
    new_stage = ame.get("stage", "observe")
    if new_stage != mode:
        _narrate(
            f"Automation level upgraded! {_short_stage(new_stage)} unlocked",
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
        parts.append(f"Approval predictor ({acc}% accurate)")
    if results.get("outcome", {}).get("trained"):
        acc = round(results["outcome"].get("metrics", {}).get("accuracy", 0) * 100)
        parts.append(f"Success predictor ({acc}% accurate)")
    if results.get("anomaly", {}).get("trained"):
        parts.append("Problem detector active")
    _narrate(
        f"AI trained on {created} historical decisions",
        "ml",
        f"Active models: {', '.join(parts) if parts else 'none yet'}. "
        f"The AI can now predict which actions your team will approve and whether they'll succeed. "
        f"These predictions appear on every recommendation going forward.",
    )

    return {"ok": True, "events_created": created, "training_results": results}


@router.post("/dashboard/api/reset")
def reset_dashboard(db=Depends(get_db)):
    """
    Full reset: clears demo cases, proposals, AME trust data, and governance gates.
    Returns the dashboard to a clean-slate state for a fresh demo.
    """
    global DEMO_ROWS, DEMO_BY_ID, DEMO_PROPOSALS, DEMO_PROPOSAL_BY_ID

    # 1. Clear in-memory demo data (instant, never blocks)
    DEMO_ROWS = []
    DEMO_BY_ID = {}
    DEMO_PROPOSALS = []
    DEMO_PROPOSAL_BY_ID = {}

    # 2. Clear AME trust tables — best-effort, skip if SQLite locks
    from src.enterprise.ame.models import AMETrustScope, AMEEvent, AMEExecution
    try:
        db.execute(AMEExecution.__table__.delete())
        db.execute(AMEEvent.__table__.delete())
        db.execute(AMETrustScope.__table__.delete())
        db.commit()
        log.info("AME trust data cleared")
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        log.warning("AME reset skipped (DB busy): %s", exc)

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
    with _feed_lock:
        ACTIVITY_FEED.clear()

    # 6. Auto-seed fresh demo cases so the board isn't empty
    try:
        seed_demo_cases(scenario_id="happy_path", seed_all=True)
        log.info("Re-seeded %d demo cases after reset", len(DEMO_ROWS))
    except Exception as exc:
        log.warning("Auto-seed after reset failed: %s", exc)

    _narrate(
        "Fresh start — demo reset and reloaded.",
        "info",
        "All data cleared and fresh scenarios loaded. "
        "Ready for a new demo — click 'Auto-Play Demo' or explore the pipeline.",
    )

    return {"ok": True, "message": "Dashboard reset and re-seeded", "cases": len(DEMO_ROWS)}

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
        industry = s.get("industry", "pharmacy")
        events = [
            {"event_type": "case_seeded", "payload": {"scenario_id": sid, "label": scenario_label, "description": s.get("description", "")}},
            {"event_type": "queue_changed", "payload": {"from": "none", "to": start_queue}},
        ]

        if industry == "insurance":
            events.append({"event_type": "claim_received", "payload": {"claim_type": "auto_adjudication", "status": "pending_review"}})
            events.append({"event_type": "policy_validated", "payload": {"policy_status": "active", "coverage": "verified"}})
        elif industry == "hr":
            events.append({"event_type": "hire_request_received", "payload": {"department": "Operations", "role": "Analyst", "start_date": "2026-02-15"}})
            events.append({"event_type": "documents_requested", "payload": {"items": ["ID verification", "Tax forms", "Direct deposit"]}})
        else:
            events.append({"event_type": "insurance_adjudicated", "payload": {"payer": "AutoPayer", "result": s.get("insurance_result", "accepted")}})

        # Add pharmacy-specific context events
        if industry == "pharmacy":
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

        # Build industry-specific tasks
        industry = s.get("industry", "pharmacy")
        if industry == "insurance":
            tasks = [
                {"name": "Intake claim submission", "assigned_to": "Claims Processor", "state": "open"},
                {"name": "Validate policy and coverage", "assigned_to": "System", "state": "pending"},
                {"name": "Assess liability and damages", "assigned_to": "Adjuster", "state": "pending"},
                {"name": "Review and approve payout", "assigned_to": "Claims Manager", "state": "pending"},
            ]
        elif industry == "hr":
            tasks = [
                {"name": "Collect new-hire documents", "assigned_to": "HR Coordinator", "state": "open"},
                {"name": "Run background check", "assigned_to": "System", "state": "pending"},
                {"name": "Provision system access and equipment", "assigned_to": "IT Ops", "state": "pending"},
                {"name": "Schedule compliance training", "assigned_to": "HR Manager", "state": "pending"},
            ]
        else:
            tasks = [{"name": "Enter NPI + patient DOB", "assigned_to": "—", "state": "open"}]
            if not s.get("refills_ok"):
                tasks.append({"name": "Request refill authorization from prescriber", "assigned_to": "Store Tech", "state": "pending"})
            if s.get("insurance_result") == "pa_required":
                tasks.append({"name": "Monitor prior authorization status", "assigned_to": "System", "state": "pending"})
            if s.get("insurance_result") == "rejected":
                tasks.append({"name": "Contact patient re: insurance update", "assigned_to": "Store Tech", "state": "pending"})

        # Industry-specific case names
        if industry == "insurance":
            case_name = f"Claims \u2022 CLM-{2000 + idx} (Demo)"
        elif industry == "hr":
            case_name = f"HR \u2022 ONB-{3000 + idx} (Demo)"
        else:
            case_name = f"Pharmacy \u2022 RX-{1000 + idx} (Demo)"

        # Industry-specific detail label (shown under case card)
        if industry == "insurance":
            detail_label = f"Claim: {s.get('insurance_result', 'accepted')}"
        elif industry == "hr":
            detail_label = "New hire onboarding"
        else:
            detail_label = f"AutoPayer: {s.get('insurance_result', 'accepted')}"

        raw = {
            "id": demo_id,
            "name": case_name,
            "state": "ACTIVE",
            "scenario_id": sid,
            "scenario_label": scenario_label,
            "industry": industry,
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
            "industry": industry,
            "insurance": detail_label,
            "tasks": len(raw["tasks"]),
            "events": len(raw["events"]),
            "is_kroger": industry == "pharmacy",
            "raw": raw,
        }

        DEMO_ROWS.append(row)
        DEMO_BY_ID[demo_id] = row
        return row

    if seed_all:
        for i, sid in enumerate(DEMO_SCENARIOS.keys(), start=1):
            _mk_case(sid, i)
        _narrate(
            f"Loaded {len(DEMO_ROWS)} cases across your pipeline.",
            "info",
            "Each case starts at a different step to show the full workflow. "
            "Every transition requires your approval — safety checkpoints are enabled. "
            "The AI starts in Learning mode: it watches your decisions before suggesting anything.",
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
        _ind = row.get("industry", "pharmacy")
        trust_pct = round(ame.get("trust", 0) * 100)
        if mode in ("guarded_auto", "conditional_auto", "full_auto"):
            gov_msg = f"AI handled this automatically ({_short_stage(mode)}, {trust_pct}% confidence)."
            if exe:
                gov_msg += f" Your team can undo this within 15 minutes if needed."
        else:
            gov_msg = f"Automation level: {_short_stage(mode)} — your team's approval was required for this step."
        _narrate(
            f"Case advanced: {case_name} moved to {_q_label(nxt, _ind)} (+30s saved)",
            "execution",
            f"{gov_msg} Each successful step earns the AI more trust toward higher automation.",
            meta={"ame_stage": mode, "trust": ame.get("trust", 0)},
        )

        # Check for stage change
        new_stage = ame.get("stage", "observe")
        if new_stage != mode:
            _narrate(
                f"Automation level upgraded! {_short_stage(new_stage)} unlocked",
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
# Auto-play Demo: run a case through the full pipeline
# =============================

@router.post("/dashboard/api/run-demo")
def run_demo_step(
    workflow_id: int = Body(..., embed=True),
    db=Depends(get_db),
):
    """
    Runs ONE step of the auto-play demo: advances the case to the next queue,
    enables the governance gate, creates proposal, approves, and executes —
    all in one call. Returns the result plus what happened for narration.

    Call repeatedly until done=true to complete the full pipeline.
    """
    row = DEMO_BY_ID.get(workflow_id)
    if not row:
        raise HTTPException(status_code=404, detail="Demo workflow not found")

    cur = row.get("queue") or "data_entry"
    t = TRANSITIONS.get(cur)
    if not t:
        _narrate(
            f"Complete! {row.get('name', 'Case')} processed through the entire pipeline.",
            "info",
            "Every step was approved, executed, and logged. The AI now has data from this case to improve future predictions. "
            "Over time, routine cases like this can be handled automatically — saving your team minutes per case.",
        )
        return {"ok": True, "done": True, "queue": cur, "message": "Case has reached the end of the pipeline."}

    to_q = t["to"]
    gk = t["gate"]
    _ind = row.get("industry", "pharmacy")

    # Step 1: Auto-authorize the safety checkpoint
    if not governance.is_authorized(gk):
        governance.authorize_gate(gk, actor="demo_autoplay", note="Auto-play demo")
        _narrate(
            f"Safety checkpoint enabled: {_q_label(cur, _ind)} \u2192 {_q_label(to_q, _ind)}",
            "gate",
            "In production, a manager would enable this pathway. The demo does it automatically so you can see the full flow.",
        )

    # Step 2: Create proposal
    pid = "auto-" + uuid.uuid4().hex[:8]
    action = {
        "type": "advance_queue",
        "label": f"Advance from {cur} to {to_q}",
        "payload": {"from_queue": cur, "to_queue": to_q},
        "confidence": 0.88,
        "safety_score": 0.93,
    }
    p = _mk_proposal(workflow_id, action)

    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=p["id"], predicted_confidence=0.88, predicted_safety=0.93,
              context={"transition": f"{cur}\u2192{to_q}", "demo_autoplay": True})

    _narrate(
        f"AI recommends: move {row.get('name', 'case')} to next step",
        "proposal",
        f"88% confident this is correct. 93% safety score. In production, your team reviews this before it executes.",
    )

    # Step 3: Approve
    now = _now_iso()
    p["status"] = "approved"
    p["approved_by"] = "Pharmacy_Manager"
    p["approved_at"] = now
    p["audit"].append({"ts": now, "event": "decision:approve",
                        "meta": {"by": "Pharmacy_Manager", "note": "Demo auto-play"}})

    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=p["id"], decision="approve", decision_by="Pharmacy_Manager",
              decision_reason="Demo auto-play approval")

    _narrate(
        f"Manager approved: move case to {_q_label(to_q, _ind)}",
        "decision",
        "Each approval teaches the AI what's safe. After enough approvals, similar cases can be auto-approved.",
    )

    # Step 4: Execute
    before_q = cur
    row["queue"] = to_q
    row["raw"]["events"].append({"event_type": "queue_changed", "payload": {"from": cur, "to": to_q}})
    row["events"] = len(row["raw"].get("events") or [])

    p["status"] = "executed"
    p["executed_at"] = _now_iso()
    p["executed_by"] = "system"
    p["audit"].append({"ts": _now_iso(), "event": "executed:advance_queue",
                        "meta": {"from": cur, "to": to_q}})

    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.EXECUTED.value, proposal_id=p["id"])
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.OUTCOME.value, proposal_id=p["id"],
              outcome_success=True, observed_time_saved_sec=45.0)

    ame = _ame_trust_info(db, cur)
    trust_pct = round(ame.get("trust", 0) * 100)

    _narrate(
        f"Done! Case moved to {_q_label(to_q, _ind)} — saved 45 seconds vs. manual processing",
        "execution",
        f"AI confidence at this step: {trust_pct}%. "
        f"Each successful case builds the AI's track record toward handling these steps automatically.",
        meta={"trust": ame.get("trust", 0)},
    )

    # Check for next step
    has_next = to_q in TRANSITIONS
    return {
        "ok": True, "done": not has_next,
        "from": before_q, "to": to_q,
        "queue": to_q,
        "trust": trust_pct,
        "ame": ame,
        "case": row,
    }


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

    /* --- Run Demo button --- */
    #runDemoBtn:disabled{opacity:.5;cursor:not-allowed}

    /* --- Toast notification --- */
    .toast{position:fixed;top:16px;right:16px;z-index:11000;background:#1e293b;border:1px solid rgba(255,255,255,.15);border-radius:10px;padding:10px 18px;color:#eaeaea;font-size:13px;font-weight:600;box-shadow:0 8px 24px rgba(0,0,0,.4);opacity:0;transform:translateY(-12px);transition:opacity .3s,transform .3s;pointer-events:none}
    .toast.show{opacity:1;transform:translateY(0)}
    .toast.success{border-left:4px solid #4ade80}
    .toast.info{border-left:4px solid #60a5fa}

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

    /* --- Responsive: Tablet --- */
    @media(max-width:960px){
      .pipeline{flex-wrap:wrap;gap:8px}
      .pipe-col{min-width:140px;max-width:none;flex:1 1 45%}
      .pipe-arrow{display:none}
      .detail-grid{grid-template-columns:1fr}
      .ml-grid{grid-template-columns:1fr 1fr}
    }

    /* --- Responsive: Mobile --- */
    @media(max-width:600px){
      body{font-size:13px;-webkit-text-size-adjust:100%}

      /* Header: stack vertically */
      header{flex-wrap:wrap;gap:6px;padding:10px 12px}
      header b{font-size:14px}
      header .muted{display:none}
      header a{margin-left:auto;font-size:11px}

      /* Controls: wrap, stack buttons */
      .controls{padding:8px 12px;gap:6px}
      .controls select{min-width:0;flex:1 1 100%;font-size:13px;padding:10px}
      .controls input{flex:1 1 100%;min-width:0;max-width:none;font-size:13px;padding:10px}
      .controls button{padding:10px 14px;font-size:12px;flex:1 1 auto}
      .controls>div[style*="width:1px"]{display:none}
      .controls>div[style*="flex:1"]{display:none}

      /* ML strip: single column */
      .ml-strip{padding:10px 12px}
      .ml-strip-header{flex-wrap:wrap;gap:6px}
      .ml-strip-title{font-size:11px}
      .ml-grid{grid-template-columns:1fr;gap:8px}
      .ml-card{padding:10px}
      .ml-card-value{font-size:16px}
      .roi-grid{grid-template-columns:repeat(4,1fr);gap:4px}
      .roi-num{font-size:15px}

      /* Activity feed */
      .feed-toggle{padding:8px 12px}
      .feed-strip.open{padding:8px 12px;max-height:180px}

      /* Pipeline: single column */
      .pipeline{padding:10px 12px;flex-direction:column;gap:8px;overflow-x:visible}
      .pipe-col{flex:1 1 100%;min-width:0;max-width:none}
      .pipe-arrow{display:none}
      .pipe-body{max-height:200px}
      .pipe-card{padding:10px 12px}
      .pipe-card-name{font-size:13px;white-space:normal}
      .pipe-header{padding:10px 12px}

      /* Detail panel */
      .detail{padding:0 12px 16px}
      .detail-bar{flex-direction:column;gap:6px;align-items:flex-start}
      .detail-actions{margin-left:0;flex-wrap:wrap;width:100%}
      .detail-actions button{flex:1 1 auto;padding:10px;font-size:12px}
      .detail-grid{grid-template-columns:1fr;gap:8px}
      .detail-pills{gap:4px}
      .detail-pills .pill{font-size:10px;padding:3px 8px}

      /* Auth checkboxes: stack */
      .auth-row{flex-direction:column;gap:8px;align-items:flex-start}
      .auth-row label{font-size:12px;padding:4px 0}

      /* Modals: full-width on mobile */
      #authModal>div{margin:2vh 8px!important;max-width:none!important;max-height:90vh!important;padding:12px!important;border-radius:12px!important}

      /* Tutorial */
      .tut-splash-card{margin:12px;padding:24px 18px;max-width:none}
      .tut-splash-card h2{font-size:17px}
      .tut-splash-card p{font-size:12px}
      .tut-tooltip{max-width:calc(100vw - 24px);min-width:0;left:12px!important;right:12px}
      .tut-tooltip h3{font-size:13px}
      .tut-tooltip p{font-size:12px}
      .tut-nav button{padding:8px 12px;font-size:12px}

      /* Toast: full width */
      .toast{left:12px;right:12px;max-width:none;font-size:12px;padding:10px 14px}

      /* Buttons: bigger tap targets */
      button{min-height:40px;padding:8px 14px}
      .tut-help-btn{width:44px;height:44px;font-size:18px}

      /* Scenario description */
      .scenario-desc{padding:6px 12px;font-size:11px}

      /* Anomaly banner */
      .anomaly-banner{padding:8px 12px;font-size:11px}

      /* Pre/JSON */
      pre{font-size:10px;padding:10px;max-height:200px}
    }

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
  <span class="muted">AI that earns your trust &bull; Every step governed &bull; Complete audit trail</span>
  <a href="/ame/dashboard" style="margin-left:auto;color:#60a5fa;text-decoration:none;font-size:12px">Trust Dashboard</a>
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
  <button id="runDemoBtn" onclick="runFullDemo()" style="background:rgba(34,197,94,.15);border-color:rgba(34,197,94,.3);color:#4ade80;font-weight:700" title="Auto-play: watch one case flow through the entire pipeline with narration (resets data)">&#9654; Auto-Play Demo</button>
  <button onclick="generateReport()" style="background:rgba(59,130,246,.12);border-color:rgba(59,130,246,.3);color:#60a5fa" title="Generate a detailed report of all decisions, proposals, and executions">&#128196; Report</button>
  <button onclick="refreshAll()">Refresh</button>
  <button onclick="resetDashboard()" style="color:#f87171;border-color:rgba(248,113,113,.3)">Reset</button>
</div>

<!-- ====== ML Intelligence Strip ====== -->
<div class="ml-strip">
  <div class="ml-strip-header">
    <span class="ml-strip-title">AI Learning Status</span>
    <button onclick="seedMlData()" style="background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.3);color:#60a5fa">Train AI on Historical Data</button>
    <button id="trainBtn" onclick="trainModels()">Retrain</button>
    <span class="muted" id="trainStatus"></span>
    <div style="flex:1"></div>
    <span class="muted" id="mlEventCount"></span>
  </div>
  <div class="ml-grid">
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotDecision"></span>Approval Predictor</div>
      <div class="ml-card-value" id="mlDecVal">Waiting for data</div>
      <div class="ml-card-sub" id="mlDecSub">Will your team approve this action? Learns from past approvals and rejections to predict the answer.</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotOutcome"></span>Success Predictor</div>
      <div class="ml-card-value" id="mlOutVal">Waiting for data</div>
      <div class="ml-card-sub" id="mlOutSub">Will this action complete successfully? Learns from past outcomes to estimate success before you commit.</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label"><span class="ml-dot off" id="dotAnomaly"></span>Problem Detector</div>
      <div class="ml-card-value" id="mlAnoVal">Waiting for data</div>
      <div class="ml-card-sub" id="mlAnoSub">Is something wrong? Monitors patterns to catch unusual rejection spikes, timing changes, or unexpected failures early.</div>
    </div>
    <div class="ml-card">
      <div class="ml-card-label">Business Impact</div>
      <div class="roi-grid">
        <div class="roi-stat"><span class="roi-num" id="roiEvents">0</span><span class="roi-label">Decisions<br/>Learned</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiAuto">0</span><span class="roi-label">Auto-<br/>approved</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiHuman">0</span><span class="roi-label">Human<br/>Review</span></div>
        <div class="roi-stat"><span class="roi-num" id="roiTime">0m</span><span class="roi-label">Time<br/>Saved</span></div>
      </div>
      <div class="muted" id="roiProjection" style="font-size:10px;margin-top:6px;text-align:center"></div>
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
        <span class="pill" id="pillQueue">step: &mdash;</span>
        <span class="pill" id="pillIns">insurance: &mdash;</span>
        <span class="pill" id="pillState">status: &mdash;</span>
        <span class="pill" id="pillAme">AI level: &mdash;</span>
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

  <div class="auth-row" id="authRow">
    <span style="font-size:11px;font-weight:700;color:var(--muted)">Safety Checkpoints:</span>
    <!-- Populated dynamically by renderDetails() based on industry -->
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
    <h2>Welcome to PactGate&trade;</h2>
    <p style="text-align:left">Your team spends <b>3-15 minutes per case</b> on routine approvals, insurance checks, and handoffs. That's hundreds of staff-hours per month.</p>
    <p style="text-align:left">PactGate <b>learns from your team's decisions</b> and progressively automates routine work &mdash; while keeping humans in control of high-stakes decisions. Every action is audited for compliance.</p>
    <p style="text-align:left;color:#4ade80;font-weight:600;font-size:12px">This demo shows a pharmacy workflow, but PactGate works for any governed process: insurance claims, HR onboarding, vendor approvals, and more.</p>
    <button class="tut-begin" onclick="tutStartWithSeed()">Start the Demo</button>
    <button class="tut-skip-link" onclick="tutDismiss()">Skip &mdash; I've seen this before</button>
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

/* ---------- Toast Notifications ---------- */
let toastEl=null;let toastTimer=null;
function showToast(msg,type){
  if(!toastEl){toastEl=document.createElement('div');toastEl.className='toast';document.body.appendChild(toastEl)}
  toastEl.textContent=msg;toastEl.className='toast '+(type||'info');
  clearTimeout(toastTimer);
  requestAnimationFrame(()=>{toastEl.classList.add('show');toastTimer=setTimeout(()=>{toastEl.classList.remove('show')},3500)});
}

async function api(path, opts={}){
  const ctrl=new AbortController();
  const timer=setTimeout(()=>ctrl.abort(),30000);
  try{
    const res=await fetch(path,{signal:ctrl.signal,headers:{"Content-Type":"application/json",...(opts.headers||{})},...opts});
    clearTimeout(timer);
    let txt="",js=null;
    try{txt=await res.text();js=txt?JSON.parse(txt):null}catch(_){}
    if(!res.ok) throw new Error((js&&(js.detail||js.message))||txt||('HTTP '+res.status));
    return js||(txt?JSON.parse(txt):{});
  }catch(e){
    clearTimeout(timer);
    if(e.name==='AbortError') throw new Error('Request timed out — server may be busy. Try again.');
    throw e;
  }
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
function stageName(s){return{observe:"Learning",propose:"Suggesting",guarded_auto:"Assisting",conditional_auto:"Semi-Auto",full_auto:"Autonomous"}[s]||s||"\u2014"}
function stageExplain(s){return{
  observe:"Level 1 \u2014 LEARNING: AI is watching your team work and learning patterns. All actions require manual approval. This is the starting point for every new workflow.",
  propose:"Level 2 \u2014 SUGGESTING: AI now recommends next steps with confidence scores. Your team still makes every decision, but AI is learning what gets approved.",
  guarded_auto:"Level 3 \u2014 ASSISTING: AI handles routine actions automatically with a 15-minute undo window. Your team can reverse any AI action within that window.",
  conditional_auto:"Level 4 \u2014 SEMI-AUTONOMOUS: AI handles cases when confidence is high enough. Near-full automation with safety guardrails still active.",
  full_auto:"Level 5 \u2014 AUTONOMOUS: AI handles this step end-to-end. Earned through consistent accuracy, alignment with your team's decisions, and proven safety record. Full audit trail maintained.",
}[s]||"No data yet \u2014 the AI hasn't observed this step. Process some cases to start building a track record."}
function trustColor(t){if(t>=.8)return"#4ade80";if(t>=.6)return"#fbbf24";if(t>=.4)return"#60a5fa";return"#6b7280"}

/* ---------- Board ---------- */
function groupByQueue(rows){
  const c={contact_manager:[],inbound_comms:[],data_entry:[],pre_verification:[],dispensing:[],verification:[]};
  rows.forEach(r=>{const q=r.queue||"unknown";if(c[q])c[q].push(r)});return c;
}
function matchesSearch(r,s){if(!s)return true;s=s.toLowerCase();return(r.name||"").toLowerCase().includes(s)||(r.queue||"").toLowerCase().includes(s)}

/* Industry-specific column labels for the pipeline */
const INDUSTRY_LABELS={
  pharmacy:{contact_manager:"Intake",inbound_comms:"Triage",data_entry:"Data Entry",pre_verification:"Verification",dispensing:"Processing",verification:"Complete"},
  insurance:{contact_manager:"Intake",inbound_comms:"Triage",data_entry:"Assessment",pre_verification:"Review",dispensing:"Approval",verification:"Settled"},
  hr:{contact_manager:"Request",inbound_comms:"Documents",data_entry:"Background Check",pre_verification:"Provisioning",dispensing:"Training",verification:"Complete"},
};

/* Detect dominant industry from current data for column labels */
/* Translate raw queue key to industry label */
function qLabel(q,ind){return(INDUSTRY_LABELS[ind||"pharmacy"]||INDUSTRY_LABELS.pharmacy)[q]||q}

function detectIndustry(){
  const counts={pharmacy:0,insurance:0,hr:0};
  ALL.forEach(r=>{const ind=r.industry||"pharmacy";if(counts[ind]!==undefined)counts[ind]++});
  if(counts.insurance>counts.pharmacy&&counts.insurance>=counts.hr)return "insurance";
  if(counts.hr>counts.pharmacy&&counts.hr>=counts.insurance)return "hr";
  return "pharmacy";
}

function getQueueOrder(){
  const keys=["contact_manager","inbound_comms","data_entry","pre_verification","dispensing","verification"];
  /* If mixed industries, show generic labels */
  const industries=new Set(ALL.map(r=>r.industry||"pharmacy"));
  let labels;
  if(industries.size<=1){
    const ind=industries.values().next().value||"pharmacy";
    labels=INDUSTRY_LABELS[ind]||INDUSTRY_LABELS.pharmacy;
  }else{
    /* Mixed: show combined labels */
    labels={contact_manager:"Intake / Request",inbound_comms:"Triage / Documents",data_entry:"Data Entry / Assessment",pre_verification:"Verification / Review",dispensing:"Processing / Approval",verification:"Complete"};
  }
  return keys.map(k=>[k,labels[k]||k]);
}

function renderBoard(){
  const s=document.getElementById("search").value.trim();
  const rows=ALL.filter(r=>matchesSearch(r,s));
  const cols=groupByQueue(rows);
  const board=document.getElementById("board");
  board.innerHTML="";

  getQueueOrder().forEach(([key,title],i)=>{
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
      const short=(r.name||"").replace(" (Demo)","");
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
  const qLabels=INDUSTRY_LABELS[wf.industry||"pharmacy"]||INDUSTRY_LABELS.pharmacy;
  document.getElementById("pillQueue").textContent="step: "+(qLabels[wf.queue]||wf.queue);
  document.getElementById("pillIns").textContent=(wf.industry==="hr"?"dept":"insurance")+": "+wf.insurance;
  document.getElementById("pillState").textContent="status: "+wf.state;

  const ame=AME_SCOPES[wf.queue]||{};
  document.getElementById("pillAme").textContent="AI: "+stageName(ame.stage)+(ame.trust>0?" ("+Math.round(ame.trust*100)+"%)":"");

  document.getElementById("json").textContent=JSON.stringify(wf.raw,null,2);

  /* tasks */
  const tEl=document.getElementById("tasks");tEl.innerHTML="";
  (wf.raw.tasks||[]).forEach(t=>{const d=document.createElement("div");d.className="item";d.innerHTML="<b>"+t.name+"</b><div class='muted'>assigned: "+(t.assigned_to||"\u2014")+" \u2022 state: "+t.state+"</div>";tEl.appendChild(d)});
  if(!(wf.raw.tasks||[]).length) tEl.innerHTML='<div class="muted">No tasks</div>';

  /* events */
  const eEl=document.getElementById("events");eEl.innerHTML="";
  (wf.raw.events||[]).slice().reverse().forEach(e=>{const d=document.createElement("div");d.className="item";d.innerHTML="<b>"+e.event_type+"</b><div class='muted'>"+JSON.stringify(e.payload||{})+"</div>";eEl.appendChild(d)});
  if(!(wf.raw.events||[]).length) eEl.innerHTML='<div class="muted">No events</div>';

  /* auth checkboxes — dynamic per industry */
  const authRow=document.getElementById("authRow");
  const ind=wf.industry||"pharmacy";
  const lbl=INDUSTRY_LABELS[ind]||INDUSTRY_LABELS.pharmacy;
  const gates=[
    {key:"kroger.prescriber_approval_to_data_entry",from:"contact_manager",to:"data_entry"},
    {key:"kroger.data_entry_to_preverify_insurance",from:"data_entry",to:"pre_verification"},
    {key:"kroger.preverify_to_access_granted",from:"pre_verification",to:"dispensing"},
  ];
  authRow.innerHTML='<span style="font-size:11px;font-weight:700;color:var(--muted)">Safety Checkpoints:</span>';
  gates.forEach((g,i)=>{
    const fromLabel=lbl[g.from]||g.from;
    const toLabel=lbl[g.to]||g.to;
    const checked=!!AUTH[g.key];
    const label=document.createElement("label");
    label.title="Enable or disable the pathway between these steps. Unchecked = cases cannot advance.";
    label.innerHTML='<input type="checkbox" id="a'+(i+1)+'" '+(checked?"checked":"")+' onchange="saveAuth(\''+g.key+'\',this.checked)"> '+fromLabel+' &rarr; '+toLabel;
    authRow.appendChild(label);
  });

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
  setStatus("Advancing case\u2026");
  try{
    const r=await api("/dashboard/api/auto-step",{method:"POST",body:JSON.stringify({workflow_id:selected.id})});
    await refreshAll();setStatus("Ready");
    if(r.from&&r.to){const si=selected?.industry||"pharmacy";showToast("Case moved: "+qLabel(r.from,si)+" \u2192 "+qLabel(r.to,si),"success")}
  }catch(e){
    const m=String(e?.message||"");
    if(m.toLowerCase().includes("not authorized")||m.includes("403")){setStatus("Enable the safety checkpoint first");showToast("Safety checkpoint not enabled for this step","info");return}
    if(m.includes("timed out")){setStatus("Server busy \u2014 try again");showToast("Server was busy. Click Auto-step again.","info");return}
    setStatus("Error");showToast("Step failed: "+m,"info");
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

/* ---------- Run Full Demo (auto-play) ---------- */
let demoRunning=false;
async function runFullDemo(){
  if(demoRunning)return;
  demoRunning=true;
  const btn=document.getElementById("runDemoBtn");
  btn.textContent="Running\u2026";btn.disabled=true;

  try{
    /* Reset (auto-seeds all scenarios) then seed a focused happy-path case */
    setStatus("Preparing demo\u2026");
    try{await api("/dashboard/api/reset",{method:"POST",body:"{}"});}catch(_){console.warn("reset slow, continuing")}
    await refreshAll();
    if(!feedOpen)toggleFeed();
    /* Pick the first happy-path case from auto-seeded data */
    let caseId=ALL.length?ALL[0].id:null;
    if(!caseId){
      const seed=await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:"happy_path",seed_all:false})});
      caseId=seed.id;
      await refreshAll();
    }

    /* Select the case */
    const wf=ALL.find(x=>x.id===caseId);
    if(wf)renderDetails(wf);
    showToast("Demo started \u2014 watch the case flow through the pipeline","info");
    await sleep(1500);

    /* Step through the pipeline */
    let stepNum=0;
    for(let step=0;step<10;step++){
      stepNum++;
      setStatus("Step "+stepNum+": advancing case\u2026");
      const r=await api("/dashboard/api/run-demo",{method:"POST",body:JSON.stringify({workflow_id:caseId})});
      await refreshAll();
      const upd=ALL.find(x=>x.id===caseId);
      if(upd)renderDetails(upd);
      if(r.from&&r.to){const di=(r.case&&r.case.industry)||"pharmacy";showToast("Step "+stepNum+": "+qLabel(r.from,di)+" \u2192 "+qLabel(r.to,di)+" (+45s saved)","success")}
      if(r.done){showToast("Demo complete! Case processed through entire pipeline.","success");break}
      await sleep(2200);
    }

    setStatus("Demo complete!");
    await sleep(1000);
    setStatus("Ready");
  }catch(e){
    console.error("runFullDemo:",e);
    setStatus("Demo error: "+e.message);
  }finally{
    demoRunning=false;
    btn.textContent="\u25b6 Auto-Play Demo";btn.disabled=false;
  }
}
function sleep(ms){return new Promise(r=>setTimeout(r,ms))}

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
    const n=dec.training_events||0;
    decVal.textContent=acc.toFixed(0)+"% accurate";
    decSub.textContent="Learned from "+n+" past team decisions. Can now predict whether your team will approve an action before they see it.";
  }else{
    dotDec.className="ml-dot off";
    decVal.textContent="Not yet trained";
    decSub.textContent="Needs 50+ team decisions to start predicting. Click 'Train AI on Historical Data' to generate training history.";
  }

  /* Outcome */
  const out=m.outcome||{};
  const dotOut=document.getElementById("dotOutcome");
  const outVal=document.getElementById("mlOutVal");
  const outSub=document.getElementById("mlOutSub");
  if(out.trained){
    dotOut.className="ml-dot on";
    const acc=((out.metrics||{}).accuracy||0)*100;
    outVal.textContent=acc.toFixed(0)+"% accurate";
    outSub.textContent="Predicts whether an action will succeed before your team commits. Prevents errors before they happen.";
  }else{
    dotOut.className="ml-dot off";
    outVal.textContent="Not yet trained";
    outSub.textContent="Needs 30+ completed actions to start predicting. Click 'Train AI on Historical Data' to generate training history.";
  }

  /* Anomaly */
  const ano=m.anomaly||{};
  const dotAno=document.getElementById("dotAnomaly");
  const anoVal=document.getElementById("mlAnoVal");
  const anoSub=document.getElementById("mlAnoSub");
  if(ano.trained){
    dotAno.className="ml-dot on";
    const pct=(ano.metrics||{}).anomaly_percentage||0;
    anoVal.textContent=pct>0?pct+"% flagged":"All clear";
    anoSub.textContent=pct>0?"Unusual patterns detected \u2014 AI will slow down automation to protect safety.":"No unusual patterns. Operations running normally.";
  }else{
    dotAno.className="ml-dot off";
    anoVal.textContent="Not yet trained";
    anoSub.textContent="Needs history to establish a baseline. Will catch unusual rejection spikes, timing changes, or failure clusters.";
  }

  /* ROI */
  document.getElementById("roiEvents").textContent=ev.total||0;
  document.getElementById("roiAuto").textContent=roi.auto_approved||0;
  document.getElementById("roiHuman").textContent=roi.human_reviewed||0;
  const mins=roi.time_saved_min||0;
  document.getElementById("roiTime").textContent=mins>=60?(mins/60).toFixed(1)+"h":mins+"m";
  document.getElementById("mlEventCount").textContent=ev.total?(ev.total+" decisions learned"):"No training data yet";

  /* ROI projection */
  const projEl=document.getElementById("roiProjection");
  if(mins>0){
    const dailyCases=200;
    const avgSavedPerCase=mins>0?(mins/(Math.max(ev.total,1))):0.5;
    const monthlyHrs=Math.round(avgSavedPerCase*dailyCases*22/60);
    const monthlySavings=monthlyHrs*25;
    projEl.textContent="Projected monthly savings at 200 cases/day: ~"+monthlyHrs+" hours ($"+monthlySavings.toLocaleString()+")";
  }else{
    projEl.textContent="Train the AI to see projected savings";
  }
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
let refreshRetries=0;
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
    refreshRetries=0;
    setStatus("Ready");
  }catch(e){
    console.error("refreshAll:",e);
    refreshRetries++;
    if(refreshRetries<3){
      setStatus("Retrying ("+refreshRetries+"/3)\u2026");
      await sleep(2000*refreshRetries);
      return refreshAll();
    }
    setStatus("Server slow \u2014 click Refresh to retry");
    refreshRetries=0;
  }
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
  if(!confirm("Reset everything? This returns the dashboard to a fresh, first-visit state."))return;
  setStatus("Resetting\u2026");
  try{
    await api("/dashboard/api/reset",{method:"POST",body:"{}"});
  }catch(e){
    console.warn("reset API slow/failed, refreshing anyway:",e);
  }

  /* Clear all UI state back to first-visit */
  selected=null;
  demoRunning=false;
  if(repTimer){clearInterval(repTimer);repTimer=null}

  /* Close panels and modals */
  document.getElementById("detailPanel").classList.remove("open");
  document.getElementById("authModal").style.display="none";
  authOpen=false;

  /* Close activity feed */
  feedOpen=false;
  document.getElementById('feedStrip').classList.remove('open');
  document.getElementById('feedArrow').classList.remove('open');

  /* Reset ML display */
  ["mlDecVal","mlOutVal","mlAnoVal"].forEach(id=>{document.getElementById(id).textContent="Waiting for data"});
  ["dotDecision","dotOutcome","dotAnomaly"].forEach(id=>{const el=document.getElementById(id);el.className="ml-dot off"});
  ["roiEvents","roiAuto","roiHuman"].forEach(id=>{document.getElementById(id).textContent="0"});
  document.getElementById("roiTime").textContent="0m";
  document.getElementById("roiProjection").textContent="";
  document.getElementById("mlEventCount").textContent="";
  document.getElementById("trainStatus").textContent="";
  ML_STATS={};AME_SCOPES={};

  /* Reset run-demo button */
  const btn=document.getElementById("runDemoBtn");
  btn.textContent="\u25b6 Auto-Play Demo";btn.disabled=false;
  const repBtn=document.getElementById("repBtn");
  if(repBtn)repBtn.textContent="Repeat: OFF";

  /* Reset search */
  document.getElementById("search").value="";

  /* Refresh data (backend re-seeded fresh cases) */
  await refreshAll();

  /* Show welcome splash as if first visit */
  localStorage.removeItem('bladnir_tut_done');
  setTimeout(tutShowSplash, 400);
  setStatus("Ready");
}

/* ================================================================
   REPORT GENERATOR
   ================================================================ */
async function generateReport(){
  setStatus("Generating report\u2026");
  let data;
  try{
    data=await api("/dashboard/api/report");
  }catch(e){
    showToast("Failed to generate report: "+e.message,"info");
    setStatus("Ready");
    return;
  }
  setStatus("Ready");

  const s=data.summary||{};
  const ts=new Date(data.generated_at||Date.now()).toLocaleString();

  let html=`<!DOCTYPE html><html><head><meta charset="utf-8"/><title>PactGate\u2122 Report \u2014 ${ts}</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',system-ui,Arial,sans-serif;background:#fff;color:#1a1a1a;font-size:13px;line-height:1.5;padding:32px 40px;max-width:1100px;margin:0 auto}
  h1{font-size:22px;margin-bottom:4px}
  h2{font-size:16px;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e5e7eb}
  h3{font-size:13px;margin:16px 0 6px;color:#374151}
  .subtitle{color:#6b7280;font-size:12px;margin-bottom:20px}
  .summary-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0 24px}
  .summary-card{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:12px;text-align:center}
  .summary-num{font-size:24px;font-weight:800;color:#111}
  .summary-label{font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px;margin-top:2px}
  table{width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:12px}
  th{background:#f3f4f6;text-align:left;padding:8px 10px;font-weight:700;border-bottom:2px solid #e5e7eb;font-size:11px;text-transform:uppercase;letter-spacing:.3px;color:#374151}
  td{padding:7px 10px;border-bottom:1px solid #f3f4f6;vertical-align:top}
  tr:hover{background:#f9fafb}
  .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase}
  .badge.approved{background:#d1fae5;color:#065f46}
  .badge.executed{background:#dbeafe;color:#1e40af}
  .badge.rejected{background:#fee2e2;color:#991b1b}
  .badge.pending{background:#fef3c7;color:#92400e}
  .trust-bar{display:inline-block;height:6px;border-radius:3px;background:#e5e7eb;width:80px;vertical-align:middle}
  .trust-fill{height:100%;border-radius:3px;background:#3b82f6}
  .audit-entry{font-size:11px;color:#6b7280;padding:2px 0}
  .section{page-break-inside:avoid}
  @media print{body{padding:16px}h2{margin-top:20px}.summary-grid{grid-template-columns:repeat(3,1fr)}}
  .footer{margin-top:32px;padding-top:12px;border-top:1px solid #e5e7eb;font-size:11px;color:#9ca3af;text-align:center}
</style></head><body>
<h1>PactGate\u2122 Workflow Report</h1>
<div class="subtitle">Generated: ${ts} &bull; Bladnir Tech LLC</div>

<div class="summary-grid">
  <div class="summary-card"><div class="summary-num">${s.total_cases||0}</div><div class="summary-label">Total Cases</div></div>
  <div class="summary-card"><div class="summary-num">${s.total_proposals||0}</div><div class="summary-label">Proposals</div></div>
  <div class="summary-card"><div class="summary-num">${s.approved||0}</div><div class="summary-label">Approved</div></div>
  <div class="summary-card"><div class="summary-num">${s.rejected||0}</div><div class="summary-label">Rejected</div></div>
  <div class="summary-card"><div class="summary-num">${s.executed||0}</div><div class="summary-label">Executed</div></div>
  <div class="summary-card"><div class="summary-num">${s.pending||0}</div><div class="summary-label">Pending</div></div>
</div>`;

  /* --- Cases --- */
  html+=`<div class="section"><h2>Cases</h2>
<table><tr><th>ID</th><th>Name</th><th>Industry</th><th>Current Step</th><th>Status</th><th>Scenario</th><th>Tasks</th></tr>`;
  (data.cases||[]).forEach(c=>{
    html+=`<tr><td>${c.id}</td><td>${esc(c.name)}</td><td>${c.industry}</td><td>${esc(c.queue)}</td><td>${c.state}</td><td>${esc(c.scenario_label)}</td><td>${(c.tasks||[]).length}</td></tr>`;
  });
  html+=`</table></div>`;

  /* --- Proposals & Decisions --- */
  html+=`<div class="section"><h2>Proposals &amp; Decisions</h2>`;
  if(!(data.proposals||[]).length){
    html+=`<p style="color:#6b7280">No proposals created yet. Run the demo to generate proposals.</p>`;
  }else{
    html+=`<table><tr><th>ID</th><th>Case</th><th>Action</th><th>Status</th><th>Proposed By</th><th>Proposed At</th><th>Approved By</th><th>Executed By</th></tr>`;
    (data.proposals||[]).forEach(p=>{
      const action=p.action||{};
      html+=`<tr>
        <td>${esc(p.id)}</td>
        <td>${esc(p.case_name)}</td>
        <td>${esc(action.label||action.type||"\u2014")}</td>
        <td><span class="badge ${p.status}">${p.status}</span></td>
        <td>${esc(p.proposed_by||"\u2014")}</td>
        <td>${p.proposed_at?new Date(p.proposed_at).toLocaleString():"\u2014"}</td>
        <td>${esc(p.approved_by||"\u2014")}</td>
        <td>${esc(p.executed_by||"\u2014")}</td>
      </tr>`;
    });
    html+=`</table>`;

    /* Detailed audit trails */
    html+=`<h3>Audit Trails</h3>`;
    (data.proposals||[]).forEach(p=>{
      if(!(p.audit||[]).length)return;
      html+=`<div style="margin-bottom:12px"><b>${esc(p.id)}</b> \u2014 ${esc(p.case_name)}`;
      (p.audit||[]).forEach(a=>{
        const meta=a.meta?(" \u2014 "+Object.entries(a.meta).map(([k,v])=>k+": "+v).join(", ")):"";
        html+=`<div class="audit-entry">${a.ts?new Date(a.ts).toLocaleTimeString():""} &bull; ${esc(a.event)}${meta}</div>`;
      });
      html+=`</div>`;
    });
  }
  html+=`</div>`;

  /* --- AI Trust Status --- */
  html+=`<div class="section"><h2>AI Trust Status</h2>`;
  if(!(data.trust_scopes||[]).length){
    html+=`<p style="color:#6b7280">No trust data recorded yet.</p>`;
  }else{
    html+=`<table><tr><th>Queue</th><th>Automation Level</th><th>Trust Score</th><th>Proposals</th><th>Executions</th><th>Override Rate</th></tr>`;
    (data.trust_scopes||[]).forEach(t=>{
      html+=`<tr>
        <td>${esc(t.queue)}</td>
        <td>${esc(t.stage)}</td>
        <td><span class="trust-bar"><span class="trust-fill" style="width:${t.trust_score}%"></span></span> ${t.trust_score}%</td>
        <td>${t.proposals}</td>
        <td>${t.executions}</td>
        <td>${t.override_rate}%</td>
      </tr>`;
    });
    html+=`</table>`;
  }
  html+=`</div>`;

  /* --- Activity Log --- */
  html+=`<div class="section"><h2>Activity Log</h2>
<table><tr><th>Time</th><th>Category</th><th>Event</th><th>Details</th></tr>`;
  (data.activity_log||[]).forEach(a=>{
    html+=`<tr>
      <td style="white-space:nowrap">${a.ts?new Date(a.ts).toLocaleTimeString():""}</td>
      <td>${esc(a.category||"")}</td>
      <td>${esc(a.message||"")}</td>
      <td style="color:#6b7280">${esc(a.detail||"")}</td>
    </tr>`;
  });
  html+=`</table></div>`;

  html+=`<div class="footer">Bladnir Tech LLC &bull; PactGate\u2122 &bull; Report generated ${ts}</div>`;
  html+=`</body></html>`;

  const w=window.open("","_blank");
  if(w){w.document.write(html);w.document.close()}
  else{showToast("Pop-up blocked \u2014 allow pop-ups for this site","info")}
}

/* ================================================================
   GUIDED TUTORIAL ENGINE
   ================================================================ */

const TUT_STEPS = [
  {
    title: "Your Operations Pipeline",
    body: "Cases flow left to right through your workflow. We loaded <b>7 real scenarios</b> \u2014 including pharmacy, insurance claims, and HR onboarding. Each column shows the AI's <b>automation level</b> for that step. Hover any badge to see what it means.",
    target: ()=>document.getElementById('board'),
    position: "bottom",
  },
  {
    title: "Every Case Has a Cost",
    body: "Each card is a case your team handles. The <b>purple labels</b> show the type: routine refill, insurance rejection, prior-auth delay. In a manual process, each of these costs <b>3\u201315 minutes of staff time</b>. Click <b>any case</b> to see its details.",
    target: ()=>document.querySelector('.pipe-card'),
    position: "bottom",
    waitForClick: true,
  },
  {
    title: "Nothing Moves Without Your Approval",
    body: "This is PactGate's core rule: <b>safety checkpoints</b> block every transition until a manager approves it. No case advances without explicit authorization. <b>Check the first box</b> to open the pathway.",
    target: ()=>document.querySelector('.auth-row'),
    position: "top",
    waitForClick: true,
  },
  {
    title: "AI Recommends, You Decide",
    body: "Click <b>Proposals</b> to see the AI's recommendation. PactGate analyzes the case and suggests the next step \u2014 but <i>your team</i> makes every decision. The AI learns from each approval or rejection.",
    target: ()=>{const btns=document.querySelectorAll('.detail-actions button');return btns[1]||null},
    position: "bottom",
    waitForClick: true,
  },
  {
    title: "Review, Approve, Execute",
    body: "Click <b>Create Proposal</b>, then <b>Approve</b>, then <b>Execute</b>. Watch the case move to the next step. Every action is logged with a full audit trail \u2014 who approved it, when, and why.",
    target: ()=>document.querySelector('#authSuggested .js-cp')||document.getElementById('authSuggested'),
    position: "left",
    waitForClick: true,
  },
  {
    title: "See What's Happening in Real Time",
    body: "The <b>Live Activity</b> feed narrates everything \u2014 approvals, AI recommendations, time saved, automation level changes. This is your operations control center.",
    target: ()=>document.getElementById('feedToggle'),
    position: "bottom",
    onArrive: ()=>{if(!feedOpen)toggleFeed()},
  },
  {
    title: "Watch Automation in Action",
    body: "Click <b>Auto-Play Demo</b> to watch a case flow through the entire pipeline automatically. Each step is narrated so you can see exactly what the AI is doing and why.",
    target: ()=>document.getElementById('runDemoBtn'),
    position: "bottom",
  },
  {
    title: "Train the AI",
    body: "Click <b>Train AI on Historical Data</b> to give the AI 80+ past decisions to learn from. Once trained, you'll see real predictions: approval likelihood, success probability, and problem detection. <b>This is where the ROI starts.</b>",
    target: ()=>document.querySelector('.ml-strip'),
    position: "bottom",
  },
  {
    title: "The Big Picture",
    body: "PactGate works for <b>any governed workflow</b>: pharmacy, insurance claims, HR onboarding, vendor approvals, expense routing. The pattern is always the same: <b>safety-first governance</b> \u2192 <b>AI that earns trust through accuracy</b> \u2192 <b>progressive automation with full audit trail</b>. Click <b>?</b> to restart this tour anytime.",
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

async function tutStartWithSeed(){
  tutHideSplash();
  setStatus("Setting up your pharmacy\u2026");
  try{
    await api("/dashboard/api/reset",{method:"POST",body:"{}"});
    await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:"happy_path",seed_all:true})});
    await refreshAll();
    if(!feedOpen)toggleFeed();
  }catch(e){console.error("tutStartWithSeed:",e)}
  setStatus("Ready");
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
    /* scroll target into view FIRST, then position after scroll settles */
    targetEl.scrollIntoView({behavior:'smooth', block:'center', inline:'nearest'});

    /* re-position after a brief delay so scroll has settled */
    const _positionTut = () => {
      const rect = targetEl.getBoundingClientRect();
      const pad = 8;
      spotlight.style.display = 'block';
      spotlight.style.top = (rect.top - pad + window.scrollY) + 'px';
      spotlight.style.left = (rect.left - pad) + 'px';
      spotlight.style.width = (rect.width + pad*2) + 'px';
      spotlight.style.height = (rect.height + pad*2) + 'px';
      backdropFill.style.display = 'none';

      const pos = step.position || 'bottom';
      const tw = 340;
      const tooltipH = 200; /* approx tooltip height */

      if(pos === 'bottom'){
        /* if tooltip would go off-screen below, flip to top */
        const bottomY = rect.bottom + pad + 12;
        if(bottomY + tooltipH > window.innerHeight && rect.top > tooltipH + 30){
          tooltip.style.top = (rect.top - pad - 12 + window.scrollY - tooltipH) + 'px';
        } else {
          tooltip.style.top = (bottomY + window.scrollY) + 'px';
        }
        tooltip.style.left = Math.max(12, Math.min(rect.left, window.innerWidth - tw - 20)) + 'px';
      } else if(pos === 'top'){
        tooltip.style.top = Math.max(12, (rect.top - pad - 12 + window.scrollY - tooltipH)) + 'px';
        tooltip.style.left = Math.max(12, Math.min(rect.left, window.innerWidth - tw - 20)) + 'px';
      } else if(pos === 'left'){
        tooltip.style.top = (rect.top + window.scrollY) + 'px';
        tooltip.style.left = Math.max(12, rect.left - tw - 20) + 'px';
      } else if(pos === 'right'){
        tooltip.style.top = (rect.top + window.scrollY) + 'px';
        tooltip.style.left = (rect.right + 16) + 'px';
      }
    };
    _positionTut();
    setTimeout(_positionTut, 400); /* re-position after smooth scroll completes */
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
  try{
    await refreshAll();
  }catch(e){
    console.error("boot refreshAll:",e);
    setStatus("Server warming up \u2014 click Refresh to retry");
    showToast("Server may be starting up. Click Refresh in a few seconds.","info");
  }
})();
</script>
</body>
</html>
    """
