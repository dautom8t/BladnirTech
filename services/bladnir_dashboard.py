


from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
from typing import Dict, Any, List
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

router = APIRouter(tags=["dashboard"])

from typing import Dict

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
    if enabled:
        governance.authorize_gate(transition_key, actor=decided_by, note=note)
    else:
        governance.revoke_gate(transition_key, actor=decided_by, note=note)
    return dashboard_get_automation()

DEMO_SCENARIOS = {
    "happy_path": {
        "label": "Happy Path",
        "insurance_result": "accepted",
        "refills_ok": True,
        "has_insurance": True,
    },
    "insurance_rejected_outdated": {
        "label": "Insurance Rejected (Outdated/Missing Info → Patient Msg)",
        "insurance_result": "rejected",
        "reject_reason": "Outdated or missing insurance information",
        "refills_ok": True,
        "has_insurance": True,
        "patient_message": {
            "type": "insurance_update_request",
            "template": "Your insurance info appears outdated or missing. Please upload/confirm your active plan to continue."
        },
    },
    "prior_auth_required": {
        "label": "Prior Authorization Required",
        "insurance_result": "pa_required",
        "refills_ok": True,
        "has_insurance": True,
        "pa": {"eta_days": 2},
        "patient_message": {
            "type": "prior_auth_notice",
            "template": "Your plan requires prior authorization. We've initiated PA; we'll update you as soon as we hear back."
        },
    },
    "no_refills_prescriber": {
        "label": "No Refills (Request to Prescriber)",
        "insurance_result": "accepted",
        "refills_ok": False,
        "has_insurance": True,
        "prescriber_request": {
            "type": "refill_request",
            "template": "No refills remaining. Refill request sent to prescriber."
        },
    },
    "no_insurance_discount_card": {
        "label": "No Insurance (Apply Discount Card)",
        "insurance_result": "no_insurance",
        "refills_ok": True,
        "has_insurance": False,
        "discount_card": {"program": "DemoRxSaver", "bin": "999999", "pcn": "DEMO", "group": "SAVER", "member": "DEMO1234"},
        "patient_message": {
            "type": "discount_card_applied",
            "template": "No active insurance found. We applied a discount card to help complete your prescription."
        },
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
                suggested.append({
                    "type": "advance_queue",
                    "label": f"Advance case to next queue from '{queue}'",
                    "payload": {"from_queue": queue, "to_queue": nq},
                    "confidence": 0.86,
                    "safety_score": 0.92,
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

    # AME: log proposal creation
    _ame_safe(db, queue=queue, action_type=action_type,
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=p["id"],
              predicted_confidence=confidence,
              predicted_safety=safety_score,
              context={"case_id": case_id, "label": label})

    return {"ok": True, "proposal": p, "ame_stage": mode}

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
        _ame_safe(db, queue=queue, action_type=action.get("type", "advance_queue"),
                  event_type=AMEEventType.PROPOSAL_DECIDED.value,
                  proposal_id=proposal_id,
                  decision=decision, decision_by=decided_by,
                  decision_reason=note or decision)

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
    return {"ok": True, "proposal": p, "case": case, "ame": ame, "ame_stage": mode}

@router.get("/dashboard/api/scenarios")
def list_demo_scenarios():
    """
    Returns available demo scenarios for the dashboard dropdown.
    """
    items = []
    for sid, s in DEMO_SCENARIOS.items():
        items.append({"id": sid, "label": s.get("label", sid)})
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

        # Pitch mode: start all scenarios at the same queue for consistent demos
        start_queue = "inbound_comms"

        raw = {
            "id": demo_id,
            "name": f"Kroger • RX-{1000 + idx} (Demo)",
            "state": "INBOUND",
            "tasks": [{"name": "Enter NPI + patient DOB", "assigned_to": "—", "state": "open"}],
            "events": [
                {"event_type": "case_seeded", "payload": {"scenario_id": sid, "label": s.get("label")}},
                {"event_type": "queue_changed", "payload": {"from": "none", "to": start_queue}},
                {"event_type": "insurance_adjudicated", "payload": {"payer": "AutoPayer", "result": s.get("insurance_result", "accepted")}},
            ],
        }

        row = {
            "id": demo_id,
            "name": raw["name"],
            "state": raw["state"],
            "queue": start_queue,
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
        return {"ok": True, "count": len(DEMO_ROWS)}
    else:
        row = _mk_case(scenario_id, 1)
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

        mode, meta = _ame_resolve_safe(db, queue=cur_queue, action_type="advance_queue")

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

        # AME: log auto-step as a complete trust cycle
        pid = f"dash-{uuid.uuid4().hex[:8]}"
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_CREATED.value,
                  proposal_id=pid, predicted_confidence=0.86, predicted_safety=0.92,
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

    # AME: log auto-step trust cycle for DB workflows
    pid = f"dash-{uuid.uuid4().hex[:8]}"
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=0.86, predicted_safety=0.92,
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
  <title>Bladnir Tech — Control Tower</title>
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

    /* --- Responsive --- */
    @media(max-width:960px){.pipeline{flex-wrap:wrap;gap:8px}.pipe-col{min-width:140px;max-width:none;flex:1 1 45%}.pipe-arrow{display:none}.detail-grid{grid-template-columns:1fr}}
  </style>
</head>
<body>

<!-- ====== Header ====== -->
<header>
  <b>Bladnir Tech &mdash; Control Tower</b>
  <span class="muted">Governed automation &bull; Queue orchestration &bull; Audit-first</span>
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

<script>
let ALL=[], AUTH={}, AME_SCOPES={}, selected=null, repTimer=null, authOpen=false;

function setStatus(t){ document.getElementById('status').textContent=t }

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
  const sel=document.getElementById("scenarioSelect");
  sel.innerHTML="";
  (d.scenarios||[]).forEach(s=>{const o=document.createElement("option");o.value=s.id;o.textContent=s.label+" ("+s.id+")";sel.appendChild(o)});
}
async function seedScenario(){
  setStatus("Seeding\u2026");
  await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:document.getElementById("scenarioSelect").value||"happy_path",seed_all:false})});
  await refreshAll();setStatus("Ready");
}
async function seedAll(){
  setStatus("Seeding all\u2026");
  await api("/dashboard/api/seed",{method:"POST",body:JSON.stringify({scenario_id:"happy_path",seed_all:true})});
  await refreshAll();setStatus("Ready");
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

    /* header */
    const hdr=document.createElement("div");hdr.className="pipe-header";
    hdr.innerHTML=
      '<div class="pipe-title">'+title+' <span style="color:var(--muted);font-weight:400">('+cnt+')</span></div>'+
      '<div class="pipe-badge" data-stage="'+stage+'">'+stageName(stage)+(trust>0?' \u2022 '+Math.round(trust*100)+'%':'')+'</div>'+
      '<div class="pipe-trust"><div class="pipe-trust-fill" style="width:'+Math.round(trust*100)+'%;background:'+trustColor(trust)+'"></div></div>';
    col.appendChild(hdr);

    /* body: case cards */
    const body=document.createElement("div");body.className="pipe-body";
    cols[key].forEach(r=>{
      const card=document.createElement("div");
      card.className="pipe-card"+(selected&&selected.id===r.id?" active":"");
      card.onclick=()=>selectCase(r.id);
      const short=(r.name||"").replace("Kroger \u2022 ","").replace(" (Demo)","");
      card.innerHTML='<div class="pipe-card-name">'+short+'</div><div class="pipe-card-ins">'+(r.insurance||"\u2014")+'</div>';
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

/* ---------- Refresh ---------- */
async function refreshAll(){
  setStatus("Loading\u2026");
  try{
    const [d1,d2]=await Promise.all([
      api("/dashboard/api/workflows").catch(e=>{console.error("workflows:",e);return{workflows:[]}}),
      api("/dashboard/api/automation").catch(e=>{console.error("automation:",e);return{authorizations:{}}})
    ]);
    ALL=d1.workflows||[];AUTH=d2.authorizations||{};
    await fetchAmeScopes();
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
      return '<div class="item"><b>'+a.label+'</b><div class="muted" style="margin-top:4px">confidence: '+(a.confidence||0).toFixed(2)+' \u2022 safety: '+(a.safety_score||0).toFixed(2)+'</div><div style="height:8px"></div><button class="js-cp" data-enc="'+enc+'">Create Proposal</button></div>';
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
      let btns="";
      if(st==="pending")btns='<button onclick="decideProposal(\\''+p.id+'\\',\\'approve\\')">Approve</button> <button onclick="decideProposal(\\''+p.id+'\\',\\'reject\\')">Reject</button>';
      else if(st==="approved")btns='<button class="primary" onclick="executeProposal(\\''+p.id+'\\')">Execute</button>';
      else btns='<span class="pill">'+st+'</span>';
      return '<div class="item"><b>'+p.id+'</b> <span class="pill" style="margin-left:6px">'+st+'</span><div class="muted" style="margin-top:4px">'+(a.label||a.type||"action")+'</div><div style="height:6px"></div><div style="display:flex;gap:6px">'+btns+'</div><div style="height:6px"></div>'+aud+'</div>';
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

/* ---------- Boot ---------- */
(async()=>{
  try{await loadScenarios()}catch(e){console.error("loadScenarios:",e)}
  try{await refreshAll()}catch(e){console.error("refreshAll:",e);setStatus("Error — check console")}
})();
</script>
</body>
</html>
    """
