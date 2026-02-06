


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
    action = {
        "type": action_type,
        "label": label,
        "payload": payload,
        "confidence": confidence,
        "safety_score": safety_score,
    }
    p = _mk_proposal(case_id, action)

    # AME: log proposal creation
    queue = case.get("queue", "unknown")
    _ame_safe(db, queue=queue, action_type=action_type,
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=p["id"],
              predicted_confidence=confidence,
              predicted_safety=safety_score,
              context={"case_id": case_id, "label": label})

    return {"ok": True, "proposal": p}

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
            # FIXED: Added error handling around governance check
            if gk:
                try:
                    governance.require_authorized(gk)
                except Exception as e:
                    p["audit"].append({
                        "ts": now,
                        "event": "execution_failed",
                        "meta": {"reason": f"governance check failed: {str(e)}"},
                    })
                    raise HTTPException(status_code=403, detail=f"Not authorized: {gk}")

            case["queue"] = to_q
            case["status"] = "demo"
            p["audit"].append({
                "ts": now,
                "event": "executed:advance_queue",
                "meta": {"from": cur, "to": case["queue"]},
            })
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
        "meta": {"by": executed_by},
    })

    # AME: log execution + successful outcome
    queue = case.get("queue", "unknown")
    act_type = action.get("type", "advance_queue")
    _ame_safe(db, queue=queue, action_type=act_type,
              event_type=AMEEventType.EXECUTED.value,
              proposal_id=proposal_id)
    _ame_safe(db, queue=queue, action_type=act_type,
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=proposal_id,
              outcome_success=True, observed_time_saved_sec=30.0)

    ame = _ame_trust_info(db, queue, act_type)
    return {"ok": True, "proposal": p, "case": case, "ame": ame}

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
    Governed auto-step for demo + DB workflows.
    Uses AUTH toggles as human-permission gates.
    """

    # ----------------
    # DEMO case (negative IDs)
    # ----------------
    if workflow_id < 0:
        row = DEMO_BY_ID.get(workflow_id)
        if not row:
            raise HTTPException(status_code=404, detail="Demo workflow not found")

        cur = row.get("queue") or "data_entry"
        nxt, gk = step_queue(cur)   # governance-enforced

        if nxt == cur:
            return {"ok": True, "note": f"No rule for queue '{cur}'", "queue": cur}

        row["raw"]["events"].append({"event_type": "auto_step", "payload": {"from": cur, "to": nxt}})
        row["raw"]["events"].append({"event_type": "queue_changed", "payload": {"from": cur, "to": nxt}})
        row["queue"] = nxt
        row["events"] = len(row["raw"].get("events") or [])

        # AME: log auto-step as a complete trust cycle
        pid = f"dash-{uuid.uuid4().hex[:8]}"
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_CREATED.value,
                  proposal_id=pid, predicted_confidence=0.86, predicted_safety=0.92,
                  predicted_time_saved_sec=30.0,
                  context={"transition": f"{cur}→{nxt}", "workflow_id": workflow_id})
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.PROPOSAL_DECIDED.value,
                  proposal_id=pid, decision="approve", decision_by="auto_step",
                  decision_reason=f"Governed auto-step {cur}→{nxt}")
        _ame_safe(db, queue=cur, action_type="advance_queue",
                  event_type=AMEEventType.OUTCOME.value,
                  proposal_id=pid, outcome_success=True, observed_time_saved_sec=30.0)

        ame = _ame_trust_info(db, cur)
        return {"ok": True, "from": cur, "to": nxt, "ame": ame}

    # ----------------
    # DB workflow (positive IDs): append events only (pilot-safe)
    # ----------------
    wf = workflow_service.get_workflow(db, workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf_read = workflow_service.to_workflow_read(wf).model_dump()
    cur = wf_read.get("queue") or "data_entry"  # FIXED: Extract actual queue instead of hardcoding
    nxt, gk = step_queue(cur)   # governance-enforced

    workflow_service.add_event(db, workflow_id, EventCreate(event_type="auto_step", payload={"from": cur, "to": nxt}))
    workflow_service.add_event(db, workflow_id, EventCreate(event_type="queue_changed", payload={"from": cur, "to": nxt}))

    # AME: log auto-step trust cycle for DB workflows
    pid = f"dash-{uuid.uuid4().hex[:8]}"
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=0.86, predicted_safety=0.92,
              predicted_time_saved_sec=30.0,
              context={"transition": f"{cur}→{nxt}", "workflow_id": workflow_id})
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=pid, decision="approve", decision_by="auto_step",
              decision_reason=f"Governed auto-step {cur}→{nxt}")
    _ame_safe(db, queue=cur, action_type="advance_queue",
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=pid, outcome_success=True, observed_time_saved_sec=30.0)

    ame = _ame_trust_info(db, cur)
    wf2 = workflow_service.get_workflow(db, workflow_id)
    return {"ok": True, "workflow": workflow_service.to_workflow_read(wf2).model_dump(), "from": cur, "to": nxt, "ame": ame}

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
      --bg:#0b0f14; --card:#111823; --line:#1f2a36; --text:#eaeaea; --muted:#9bb0c5;
      --btn:#ffffff; --btnText:#0b0f14;
      --pill:#0d131c;
    }
    body{margin:0;font-family:Arial;background:var(--bg);color:var(--text);}
    header{display:flex;align-items:center;gap:12px;padding:16px 18px;border-bottom:1px solid var(--line);}
    header b{font-size:16px;letter-spacing:.2px}
    .muted{color:var(--muted);font-size:12px}
    .wrap{display:grid;grid-template-columns: 420px 1fr; height: calc(100vh - 57px);}
    .left{border-right:1px solid var(--line); overflow:auto; padding:14px;}
    .right{overflow:auto; padding:14px;}
    .card{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px; margin-bottom:12px;}
    .row{display:flex;gap:10px;flex-wrap:wrap}
    .pill{background:var(--pill); border:1px solid var(--line); padding:6px 10px; border-radius:999px; font-size:12px; color:var(--muted);}
    button{cursor:pointer;border-radius:12px;padding:10px 12px;border:1px solid var(--line); background:transparent; color:var(--text);}
    button.primary{background:var(--btn);color:var(--btnText);border-color:var(--btn);font-weight:800}
    button.small{padding:8px 10px;font-size:12px;border-radius:10px}
    input{width:100%;padding:10px;border-radius:12px;border:1px solid var(--line);background:var(--pill);color:var(--text);box-sizing:border-box}
    select{width:100%;padding:10px;border-radius:12px;border:1px solid var(--line);background:var(--pill);color:var(--text);box-sizing:border-box}
    /* FIXED: Updated to 6 columns to match 6 queues */
    .cols{display:grid;grid-template-columns: repeat(6, 1fr); gap:12px;}
    .col{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:12px;min-height:220px}
    .col h3{margin:0 0 8px 0;font-size:13px;color:var(--muted);font-weight:700}
    .wf{border:1px solid var(--line);border-radius:14px;padding:10px;background:var(--pill);margin-bottom:10px;cursor:pointer}
    .wf b{display:block;margin-bottom:6px;font-size:13px}
    .meta{display:flex;gap:8px;flex-wrap:wrap}
    .kv{font-size:12px;color:var(--muted)}
    .split{display:grid;grid-template-columns: 1fr 1fr; gap:12px;}
    .item{border:1px solid var(--line);border-radius:14px;padding:10px;background:var(--pill);margin-bottom:10px}
    pre{white-space:pre-wrap;word-break:break-word;background:#0b0b0b;border-radius:14px;padding:12px;border:1px solid #121a22;color:#c6ff9a}
    @media (max-width:1100px){ .wrap{grid-template-columns:1fr} .left{border-right:none;border-bottom:1px solid var(--line)} .cols{grid-template-columns:1fr 1fr} .split{grid-template-columns:1fr} }
  </style>
</head>
<body>
<header>
  <b>Bladnir Tech — Control Tower</b>
  <span class="muted">Governed automation • Queue orchestration • Audit-first</span>
  <a href="/ame/dashboard" style="margin-left:auto;color:#60a5fa;text-decoration:none;font-size:13px">AME Trust Dashboard</a>
  <a href="/kroger" style="color:#60a5fa;text-decoration:none;font-size:13px">Kroger Demo</a>
  <span class="muted" id="status">Loading…</span>
</header>

<div class="wrap">
  <div class="left">
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <b style="font-size:14px">Queues</b>
          <div class="muted">Click a case card to inspect • Use auto-step after authorization</div>
        </div>
        <div class="row">
          <button class="small" onclick="refreshAll()">Refresh</button>
        </div>
      </div>

      <div style="height:12px"></div>

      <!-- Scenario Dropdown + Seed Buttons -->
      <div class="row" style="align-items:center">
        <div style="flex:1; min-width:220px">
          <div class="muted" style="margin-bottom:6px">Demo Scenario</div>
          <select id="scenarioSelect"></select>
        </div>
        <div class="row" style="align-items:flex-end">
          <button class="small" onclick="seedScenario()">Seed Scenario</button>
          <button class="small" onclick="seedAll()">Seed All</button>
        </div>
      </div>

      <div style="height:12px"></div>

      <input id="search" placeholder="Search cases by name/queue…" oninput="renderBoard()" />
    </div>

    <div class="cols" id="board"></div>
  </div>

  <div class="right">
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <b style="font-size:14px">Case Details</b>
          <div class="muted" id="caseMeta">No case selected.</div>
        </div>
        <div class="row">
          <button class="small" onclick="autoStep()">Auto-step</button>
          <button class="small" onclick="openAuthModal()">Proposals</button>
          <button class="small" onclick="toggleJson()">Toggle JSON</button>
        </div>
      </div>

      <div style="height:10px"></div>

      <!-- Repetition toggle -->
      <div class="row" style="align-items:center; justify-content:space-between">
        <div class="row" style="align-items:center">
          <button class="small" id="repBtn" onclick="toggleRepetition()">Repetition: OFF</button>
          <label class="pill" style="display:flex; align-items:center; gap:8px">
            <input type="checkbox" id="repeatTasks">
            Repeat tasks
          </label>
        </div>
        <span class="muted">Adds tick events (and optionally tasks) on a loop</span>
      </div>

      <div style="height:10px"></div>

      <div class="row">
        <span class="pill" id="pillQueue">queue: —</span>
        <span class="pill" id="pillIns">insurance: —</span>
        <span class="pill" id="pillState">state: —</span>
      </div>

      <div style="height:12px"></div>

      <div class="card" style="margin-bottom:0">
        <b style="font-size:13px">Authorized Automation</b>
        <div class="muted" style="margin-top:6px">Automation is OFF by default. Enable only with human permission.</div>
        <div style="height:10px"></div>
        <div class="row">
          <label class="pill"><input type="checkbox" id="a1" onchange="saveAuth('kroger.prescriber_approval_to_data_entry', this.checked)"> Prescriber→Data Entry</label>
          <label class="pill"><input type="checkbox" id="a2" onchange="saveAuth('kroger.data_entry_to_preverify_insurance', this.checked)"> Data Entry→Pre-Verify + Insurance</label>
          <label class="pill"><input type="checkbox" id="a3" onchange="saveAuth('kroger.preverify_to_access_granted', this.checked)"> Pre-Verify→Cleared</label>
        </div>
      </div>
    </div>

    <div class="split">
      <div class="card">
        <b style="font-size:13px">Tasks</b>
        <div id="tasks"></div>
      </div>

      <div class="card">
        <b style="font-size:13px">Timeline</b>
        <div id="events"></div>
      </div>
    </div>

    <pre id="json" style="display:none">{}</pre>
  </div>
</div>

<script>
  let ALL = [];
  let AUTH = {};
  let selected = null;
  
  let repTimer = null; // repetition loop timer
  
  function setStatus(t){ document.getElementById('status').textContent = t; }
  
  async function api(path, opts={}){
    const res = await fetch(path, {
      headers: { "Content-Type":"application/json", ...(opts.headers||{}) },
      ...opts
    });
    
    // Try to extract a meaningful message
    let bodyText = "";
    let bodyJson = null;
    try {
      bodyText = await res.text();
      bodyJson = bodyText ? JSON.parse(bodyText) : null;
    } catch (_) {
      // ignore parse errors
    }
    
    if(!res.ok){
      const detail =
        (bodyJson && (bodyJson.detail || bodyJson.message)) ||
        bodyText ||
        `HTTP ${res.status}`;
      throw new Error(detail);
    }
    
    return bodyJson || (bodyText ? JSON.parse(bodyText) : {});
  }
  
  function toggleJson(){
    const el = document.getElementById("json");
    el.style.display = (el.style.display === "none") ? "block" : "none";
  }
  
  async function loadScenarios(){
    const d = await api("/dashboard/api/scenarios");
    const sel = document.getElementById("scenarioSelect");
    sel.innerHTML = "";
    (d.scenarios || []).forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label + " (" + s.id + ")";
      sel.appendChild(opt);
    });
  }
  
  async function seedScenario(){
    const sel = document.getElementById("scenarioSelect");
    const scenario_id = sel.value || "happy_path";
    setStatus("Seeding scenario…");
    await api("/dashboard/api/seed", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ scenario_id, seed_all: false })
    });
    await refreshAll();
    setStatus("Ready");
  }
  
  async function seedAll(){
    setStatus("Seeding all scenarios…");
    await api("/dashboard/api/seed", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ scenario_id: "happy_path", seed_all: true })
    });
    await refreshAll();
    setStatus("Ready");
  }
  
  function groupByQueue(rows){
    const cols = {
      contact_manager: [],
      inbound_comms: [],
      data_entry: [],
      pre_verification: [],
      dispensing: [],
      verification: [],
    };
    
    rows.forEach(r => {
      const q = r.queue || "unknown";
      if(cols[q]) cols[q].push(r);
    });
    return cols;
  }
  
  function matchesSearch(r, s){
    if(!s) return true;
    s = s.toLowerCase();
    return (r.name||"").toLowerCase().includes(s) || (r.queue||"").toLowerCase().includes(s);
  }
  
  function renderBoard(){
    const s = document.getElementById("search").value.trim();
    const rows = ALL.filter(r => matchesSearch(r, s));
    
    const cols = groupByQueue(rows);
    const board = document.getElementById("board");
    board.innerHTML = "";
    
    const order = [
      ["contact_manager", "Contact Manager"],
      ["inbound_comms", "Inbound Comms"],
      ["data_entry", "Data Entry"],
      ["pre_verification", "Pre-Verification"],
      ["dispensing", "Dispensing"],
      ["verification", "Verification"],
    ];
    
    order.forEach(([key, title]) => {
      const col = document.createElement("div");
      col.className = "col";
      col.innerHTML = `<h3>${title} <span class="muted">(${cols[key].length})</span></h3>`;
      cols[key].forEach(r => {
        const div = document.createElement("div");
        div.className = "wf";
        div.onclick = () => selectCase(r.id);
        div.innerHTML = `
          <b>#${r.id} • ${r.name}</b>
          <div class="meta">
            <span class="kv">state: ${r.state}</span>
            <span class="kv">tasks: ${r.tasks}</span>
            <span class="kv">events: ${r.events}</span>
          </div>
          <div class="kv" style="margin-top:6px">${r.insurance}</div>
        `;
        col.appendChild(div);
      });
      board.appendChild(col);
    });
  }
  
  function renderDetails(wf){
    selected = wf;
    document.getElementById("caseMeta").textContent = `#${wf.id} • ${wf.name}`;
    document.getElementById("pillQueue").textContent = `queue: ${wf.queue}`;
    document.getElementById("pillIns").textContent = `insurance: ${wf.insurance}`;
    document.getElementById("pillState").textContent = `state: ${wf.state}`;
    document.getElementById("json").textContent = JSON.stringify(wf.raw, null, 2);
    
    // tasks
    const tasksEl = document.getElementById("tasks");
    tasksEl.innerHTML = "";
    (wf.raw.tasks||[]).forEach(t => {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `<b>${t.name}</b><div class="muted">assigned: ${t.assigned_to || "—"} • state: ${t.state}</div>`;
      tasksEl.appendChild(div);
    });
    
    // events
    const evEl = document.getElementById("events");
    evEl.innerHTML = "";
    const events = (wf.raw.events||[]).slice().reverse();
    events.forEach(e => {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `<b>${e.event_type}</b><div class="muted">${JSON.stringify(e.payload||{})}</div>`;
      evEl.appendChild(div);
    });
    
    // set checkboxes
    document.getElementById("a1").checked = !!AUTH["kroger.prescriber_approval_to_data_entry"];
    document.getElementById("a2").checked = !!AUTH["kroger.data_entry_to_preverify_insurance"];
    document.getElementById("a3").checked = !!AUTH["kroger.preverify_to_access_granted"];
  }
  
  async function selectCase(id){
    const wf = ALL.find(x => x.id === id);
    if(!wf) return;
    renderDetails(wf);
  }
  
  async function saveAuth(key, enabled){
    setStatus("Saving authorization…");
    const res = await api("/dashboard/api/automation", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ transition_key: key, enabled })
    });
    AUTH = res.authorizations || {};
    setStatus("Ready");
  }
  
  async function autoStep(){
    if(!selected) return alert("Select a case first.");
    setStatus("Auto-stepping…");
    
    try{
      await api("/dashboard/api/auto-step", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ workflow_id: selected.id })
      });
      
      await refreshAll();
      setStatus("Ready");
    } catch(e){
      const msg = String(e?.message || "");
      console.error("autoStep error:", e);
      
      // FIXED: More robust governance error detection
      if(msg.toLowerCase().includes("not authorized") || msg.toLowerCase().includes("403")){
        setStatus("Requires approval");
        return;
      }
      
      setStatus("Error");
      alert(`Auto-step failed: ${msg}`);
    }
  }
  
  async function simulateOnce(){
    if(!selected) return;
    const repeat_tasks = document.getElementById("repeatTasks").checked;
    await api("/dashboard/api/simulate", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ workflow_id: selected.id, cycles: 1, repeat_tasks })
    });
    await refreshAll();
  }
  
  function toggleRepetition(){
    const btn = document.getElementById("repBtn");
    
    if(repTimer){
      clearInterval(repTimer);
      repTimer = null;
      btn.textContent = "Repetition: OFF";
      setStatus("Ready");
      return;
    }
    
    if(!selected){
      alert("Select a case first.");
      return;
    }
    
    btn.textContent = "Repetition: ON";
    setStatus("Repetition running…");
    
    // every 3 seconds, add a tick event (and optionally tasks)
    repTimer = setInterval(() => {
      simulateOnce().catch(err => {
        console.error(err);
        clearInterval(repTimer);
        repTimer = null;
        btn.textContent = "Repetition: OFF";
        setStatus("Ready");
      });
    }, 3000);
  }
  
  async function refreshAll(){
    setStatus("Loading…");
    
    const d1 = await api("/dashboard/api/workflows");
    const d2 = await api("/dashboard/api/automation");
    
    ALL = d1.workflows || [];
    AUTH = d2.authorizations || {};
    
    renderBoard();
    
    if(selected){
      const wf = ALL.find(x => x.id === selected.id);
      if(wf) renderDetails(wf);
    }
    
    if(authOpen){
      refreshAuthModal().catch(()=>{});
    }
    
    setStatus("Ready");
  }
  
  // -------------------------
  // Proposal Modal (Authorized Automation)
  // -------------------------
  let authOpen = false;
  
  function closeAuthModal(){
    authOpen = false;
    document.getElementById("authModal").style.display = "none";
  }
  
  async function openAuthModal(){
    if(!selected) return alert("Select a case first.");
    authOpen = true;
    document.getElementById("authModal").style.display = "block";
    await refreshAuthModal();
  }
  
  async function refreshAuthModal(){
    if(!selected) return;
    const caseId = selected.id;
    
    const d = await api(`/dashboard/api/cases/${caseId}`);
    const c = d.case || {};
    document.getElementById("authMeta").textContent = `Case #${c.id} • ${c.name || ""} • queue: ${c.queue}`;
    
    // Suggested actions
    const sug = d.suggested_actions || [];
    const sugEl = document.getElementById("authSuggested");
    if(!sug.length){
      sugEl.innerHTML = `<div class="muted">No suggestions available.</div>`;
    } else {
      sugEl.innerHTML = sug.map(a => {
        const conf = (a.confidence ?? 0.0).toFixed(2);
        const safe = (a.safety_score ?? 0.0).toFixed(2);
        const payloadStr = JSON.stringify(a.payload || {});
        
        const encoded = encodeURIComponent(JSON.stringify(a));
        
        return `
          <div class="item">
            <b>${a.label}</b>
            <div class="muted" style="margin-top:6px">type: ${a.type} • confidence: ${conf} • safety: ${safe}</div>
            <div class="muted" style="margin-top:6px">payload: ${payloadStr}</div>
            <div style="height:10px"></div>
            <button class="small js-create-proposal" data-encoded="${encoded}">Create Proposal</button>
          </div>
        `;
      }).join("");
      
      sugEl.querySelectorAll(".js-create-proposal").forEach(btn => {
        btn.onclick = () => {
          const encoded = btn.getAttribute("data-encoded") || "";
          const a = JSON.parse(decodeURIComponent(encoded));
          createProposal(a);
        };
      });
    }
    
    // Proposals
    const props = d.proposals || [];
    const pEl = document.getElementById("authProposals");
    if(!props.length){
      pEl.innerHTML = `<div class="muted">No proposals yet for this case.</div>`;
    } else {
      pEl.innerHTML = props.map(p => {
        const a = p.action || {};
        const status = p.status || "unknown";
        const audit = (p.audit || []).slice().reverse().slice(0,4);
        const auditHtml = audit.map(x => `<div class="muted">• ${x.ts}: ${x.event}</div>`).join("");
        
        let buttons = "";
        if(status === "pending"){
          buttons = `
            <button class="small" onclick="decideProposal('${p.id}','approve')">Approve</button>
            <button class="small" onclick="decideProposal('${p.id}','reject')">Reject</button>
          `;
        } else if(status === "approved"){
          buttons = `<button class="small primary" onclick="executeProposal('${p.id}')">Execute</button>`;
        } else {
          buttons = `<span class="pill">status: ${status}</span>`;
        }
        
        return `
          <div class="item">
            <b>${p.id}</b> <span class="pill" style="margin-left:8px">${status}</span>
            <div class="muted" style="margin-top:6px">${a.label || a.type || "action"}</div>
            <div style="height:8px"></div>
            <div class="row">${buttons}</div>
            <div style="height:10px"></div>
            <div class="muted"><b>Audit (latest):</b></div>
            ${auditHtml || `<div class="muted">—</div>`}
          </div>
        `;
      }).join("");
    }
  }
  
  async function createProposal(a){
    if(!selected) return;
    const body = {
      action_type: a.type,
      label: a.label,
      payload: a.payload || {},
      confidence: a.confidence ?? 0.75,
      safety_score: a.safety_score ?? 0.75,
    };
    await api(`/dashboard/api/cases/${selected.id}/propose`, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    await refreshAll();
    await refreshAuthModal();
  }
  
  async function decideProposal(pid, decision){
    // FIXED: Made the decided_by value configurable (though still hardcoded for now)
    const decidedBy = "Pharmacy_Manager"; // Could be made into a user input field
    await api(`/dashboard/api/automation/${pid}/decide`, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ decision, decided_by: decidedBy, note: "" })
    });
    await refreshAll();
    await refreshAuthModal();
  }
  
  async function executeProposal(pid){
    try{
      await api(`/dashboard/api/automation/${pid}/execute`, {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ executed_by: "system" })
      });
    } catch(e){
      alert(e.message);
    }
    await refreshAll();
    await refreshAuthModal();
  }
  
  // boot
  (async () => {
    await loadScenarios();
    await refreshAll();
  })();
</script>

<!-- =========================
     Authorized Automation Modal
     ========================= -->
<div id="authModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.65); z-index:9999;">
  <div style="max-width:920px; margin:5vh auto; background:#0f1622; color:#fff; border-radius:16px; border:1px solid rgba(255,255,255,0.12); padding:16px;">
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
      <div>
        <div style="font-weight:900; font-size:14px;">Authorized Automation — Proposal Review</div>
        <div class="muted" id="authMeta" style="margin-top:4px;">—</div>
      </div>
      <div class="row">
        <button class="small" onclick="refreshAuthModal()">Refresh</button>
        <button class="small" onclick="closeAuthModal()">Close</button>
      </div>
    </div>

    <div style="height:12px"></div>

    <div class="split">
      <div class="card" style="margin:0">
        <b style="font-size:13px">Suggested Actions</b>
        <div style="height:10px"></div>
        <div id="authSuggested"></div>
      </div>

      <div class="card" style="margin:0">
        <b style="font-size:13px">Proposals & Audit</b>
        <div style="height:10px"></div>
        <div id="authProposals"></div>
      </div>
    </div>
  </div>
</div>

</body>
</html>
    """
