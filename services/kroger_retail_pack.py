"""
Kroger Retail Pharmacy Pack (Demo)

Drop-in module that adds:
- Kroger retail scenario templates (refill request starts BEFORE Rx is received)
- Integration stub endpoints to simulate prescriber approval + queue movement
- A lightweight Kroger demo UI at /kroger for click-through scenarios

Render-friendly: no external services required.

How to use:
1) Save this file as: services/kroger_retail_pack.py
2) In main.py add:
      from services.kroger_retail_pack import router as kroger_router
      app.include_router(kroger_router)
3) Redeploy / restart.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends
from fastapi.responses import HTMLResponse

from models.database import get_db
from models.schemas import (
    EventCreate,
    TaskCreate,
    TaskState,
    WorkflowCreate,
    WorkflowRead,
    WorkflowState,
)
from services import workflow as workflow_service
from src.enterprise.ame.service import log_event as _ame_log_event
from src.enterprise.ame.models import AMEEventType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["kroger"])


# -------------------------
# AME Trust Integration
# -------------------------

_AME_SITE = "kroger_demo"


def _ame_safe(db, **kwargs):
    """Log AME trust event. Non-blocking: failures are logged but never break the workflow."""
    try:
        return _ame_log_event(db, tenant_id="default", site_id=_AME_SITE, **kwargs)
    except Exception as exc:
        logger.warning("AME event skipped: %s", exc)
        return None


# -------------------------
# Scenario Templates
# -------------------------

KROGER_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "kroger_refill_atorvastatin": {
        "name": "Kroger Retail — Atorvastatin Refill Authorization",
        "description": (
            "Refill Request (Outbound) → Await Prescriber → Authorization/Rx Received → "
            "Data Entry Queue → (Push to Pre-Verification runs Insurance) → Pharmacist Review"
        ),
        # Starts BEFORE Rx is received
        "seed_tasks": [
            {"name": "Create Contact Manager Item (Refill Pending)", "assigned_to": "Store Tech"},
            {"name": "Send Refill Request to Prescriber Office", "assigned_to": "Store Tech"},
            {"name": "Await Prescriber Response (SLA)", "assigned_to": "System"},
        ],
        "seed_events": [
            {
                "event_type": "refill_request_initiated",
                "payload": {"queue": "contact_manager", "rx_status": "not_received"},
            },
            {
                "event_type": "refill_request_sent",
                "payload": {"to": "prescriber_office", "method": "eRx/fax/phone", "rx_status": "not_received"},
            },
        ],
        # Prefill data-entry defaults for the UI
        "defaults": {
            "drug": "atorvastatin",
            "sig": "Take 1 tablet by mouth daily",
            "days_supply": 90,
        },
    }
}


def _add_event(db, workflow_id: int, event_type: str, payload: Optional[dict] = None) -> None:
    workflow_service.add_event(
        db,
        workflow_id,
        EventCreate(event_type=event_type, payload=payload or {}),
    )


def _add_task(db, workflow_id: int, name: str, assigned_to: Optional[str] = None):
    return workflow_service.add_task(
        db,
        workflow_id,
        TaskCreate(name=name, assigned_to=assigned_to),
    )


# -------------------------
# API: list scenarios
# -------------------------

@router.get("/kroger/scenarios")
def list_kroger_scenarios():
    return [
        {
            "key": k,
            "name": v["name"],
            "description": v.get("description"),
            "defaults": v.get("defaults", {}),
        }
        for k, v in KROGER_SCENARIOS.items()
    ]


# -------------------------
# API: start scenario (outbound refill request BEFORE Rx exists)
# -------------------------

@router.post("/kroger/start", response_model=WorkflowRead)
def kroger_start_scenario(
    scenario_key: str = Body(..., embed=True),
    store_id: Optional[str] = Body(None, embed=True),
    patient_ref: Optional[str] = Body(None, embed=True),
    contact_method: str = Body("phone", embed=True),  # phone | fax | eRx | sms
    db=Depends(get_db),
):
    if scenario_key not in KROGER_SCENARIOS:
        raise ValueError(f"Unknown Kroger scenario: {scenario_key}")

    sc = KROGER_SCENARIOS[scenario_key]
    name = sc["name"]
    if store_id:
        name = f"{name} (Store {store_id})"

    wf_in = WorkflowCreate(
        name=name,
        description=sc.get("description"),
        initial_tasks=[TaskCreate(**t) for t in sc.get("seed_tasks", [])],
    )
    wf = workflow_service.create_workflow(db, wf_in)

    # Seed events (starts BEFORE Rx is received)
    for e in sc.get("seed_events", []):
        payload = dict(e.get("payload") or {})
        payload.update({"store_id": store_id, "patient_ref": patient_ref})
        _add_event(db, wf.id, e["event_type"], payload)

    # Record contact method used (makes it feel like Contact Manager)
    _add_event(
        db,
        wf.id,
        "contact_event_created",
        {"method": contact_method, "queue": "contact_manager", "store_id": store_id},
    )

    wf = workflow_service.get_workflow(db, wf.id)
    return workflow_service.to_workflow_read(wf)


# -------------------------
# API: prescriber approves refill (THIS is when Rx is received)
# Then move Rx to Data Entry queue + delete from Contact Manager queue.
# -------------------------

@router.post("/kroger/prescriber-approval", response_model=WorkflowRead)
def kroger_prescriber_approval(
    workflow_id: int = Body(..., embed=True),
    prescriber_office: str = Body("Prescriber Office", embed=True),
    method: str = Body("eRx", embed=True),  # eRx | fax | phone
    rx_ref: Optional[str] = Body(None, embed=True),
    db=Depends(get_db),
):
    # External event: approval received
    _add_event(
        db,
        workflow_id,
        "prescriber_refill_approved",
        {"prescriber_office": prescriber_office, "method": method, "rx_ref": rx_ref},
    )

    # This is effectively "Rx received" for processing
    _add_event(
        db,
        workflow_id,
        "authorization_received",
        {"rx_ref": rx_ref, "rx_status": "received"},
    )

    # Move to Data Entry queue
    _add_event(
        db,
        workflow_id,
        "queue_changed",
        {"from": "contact_manager", "to": "data_entry"},
    )

    # Requirement: once Rx moves into Data Entry, delete request from Contact Manager queue
    _add_event(
        db,
        workflow_id,
        "contact_manager_removed",
        {"reason": "moved_to_data_entry"},
    )

    # Optional: create explicit operational task in case you want it visible
    _add_task(db, workflow_id, "Data Entry: Enter SIG / Qty / Days Supply", "Data Entry")

    # --- AME trust tracking: prescriber approval is a successful queue advance ---
    pid = f"kroger-{uuid.uuid4().hex[:8]}"
    _ame_safe(db, queue="contact_manager", action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=0.88, predicted_safety=0.95,
              predicted_time_saved_sec=120.0,
              features={"method": method, "rx_ref": rx_ref},
              context={"transition": "contact_manager→data_entry"})
    _ame_safe(db, queue="contact_manager", action_type="advance_queue",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=pid, decision="approve", decision_by="prescriber",
              decision_reason="Rx received and approved")
    _ame_safe(db, queue="contact_manager", action_type="advance_queue",
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=pid, outcome_success=True, observed_time_saved_sec=90.0)

    wf = workflow_service.get_workflow(db, workflow_id)
    return workflow_service.to_workflow_read(wf)


# -------------------------
# API: Push from Data Entry to Pre-Verification (runs insurance during push)
# This matches: "Run insurance happens in the process of pushing rx
# from data entry queue to preverification queue"
# -------------------------

@router.post("/kroger/submit-preverification", response_model=WorkflowRead)
def kroger_submit_preverification(
    workflow_id: int = Body(..., embed=True),
    sig: str = Body(..., embed=True),
    days_supply: int = Body(90, embed=True),
    payer: str = Body("Unknown", embed=True),
    insurance_result: str = Body("accepted", embed=True),  # accepted | rejected
    db=Depends(get_db),
):
    # Mark data entry completed (event)
    _add_event(
        db,
        workflow_id,
        "data_entry_completed",
        {"sig": sig, "days_supply": days_supply},
    )

    # Run insurance as part of the push
    _add_event(
        db,
        workflow_id,
        "insurance_adjudicated",
        {"payer": payer, "result": insurance_result},
    )

    is_accepted = insurance_result == "accepted"

    if is_accepted:
        # Move into Pre-Verification queue for pharmacist review
        _add_event(
            db,
            workflow_id,
            "queue_changed",
            {"from": "data_entry", "to": "pre_verification"},
        )
        _add_task(db, workflow_id, "Pharmacist Pre-Verification Review", "Pharmacist")
    else:
        # Rejection path (exception work)
        _add_event(
            db,
            workflow_id,
            "queue_changed",
            {"from": "data_entry", "to": "rejection_resolution"},
        )
        _add_task(db, workflow_id, "Resolve Insurance Rejection", "Store Tech")
        _add_task(db, workflow_id, "Patient Outreach (rejection)", "Store Tech")

    # --- AME trust tracking: insurance adjudication outcome ---
    pid = f"kroger-{uuid.uuid4().hex[:8]}"
    _ame_safe(db, queue="data_entry", action_type="insurance_adjudication",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=0.82, predicted_safety=0.90,
              predicted_time_saved_sec=180.0,
              features={"payer": payer, "sig": sig, "days_supply": days_supply},
              context={"transition": "data_entry→pre_verification"})
    _ame_safe(db, queue="data_entry", action_type="insurance_adjudication",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=pid,
              decision="approve" if is_accepted else "reject",
              decision_by="system",
              decision_reason=f"Insurance {insurance_result}")
    _ame_safe(db, queue="data_entry", action_type="insurance_adjudication",
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=pid, outcome_success=is_accepted,
              observed_error=not is_accepted,
              observed_time_saved_sec=150.0 if is_accepted else 0.0)

    wf = workflow_service.get_workflow(db, workflow_id)
    return workflow_service.to_workflow_read(wf)


# -------------------------
# API: Pharmacist pre-verification result
# -------------------------

@router.post("/kroger/pharmacist-preverify", response_model=WorkflowRead)
def kroger_pharmacist_preverify(
    workflow_id: int = Body(..., embed=True),
    decision: str = Body("approved", embed=True),  # approved | rejected
    notes: Optional[str] = Body(None, embed=True),
    db=Depends(get_db),
):
    _add_event(
        db,
        workflow_id,
        "pre_verification_reviewed",
        {"decision": decision, "notes": notes},
    )

    wf = workflow_service.get_workflow(db, workflow_id)
    is_approved = decision == "approved"
    if wf and is_approved:
        # For demo purposes, you can optionally reflect progress in workflow state:
        # (Keep your enum as-is; DISPENSED is a reasonable "ready for fill" proxy in demo.)
        wf.state = WorkflowState.ACCESS_GRANTED  # indicates cleared/ready in your state machine
        wf.update_timestamp()
        db.commit()
        db.refresh(wf)

    # --- AME trust tracking: pharmacist clinical decision ---
    pid = f"kroger-{uuid.uuid4().hex[:8]}"
    _ame_safe(db, queue="pre_verification", action_type="pharmacist_review",
              event_type=AMEEventType.PROPOSAL_CREATED.value,
              proposal_id=pid, predicted_confidence=0.85, predicted_safety=0.92,
              predicted_time_saved_sec=60.0,
              context={"transition": "pre_verification→dispensing"})
    _ame_safe(db, queue="pre_verification", action_type="pharmacist_review",
              event_type=AMEEventType.PROPOSAL_DECIDED.value,
              proposal_id=pid,
              decision="approve" if is_approved else "reject",
              decision_by="pharmacist",
              decision_reason=notes or ("Cleared for dispensing" if is_approved else "Rejected"))
    _ame_safe(db, queue="pre_verification", action_type="pharmacist_review",
              event_type=AMEEventType.OUTCOME.value,
              proposal_id=pid, outcome_success=is_approved,
              observed_time_saved_sec=45.0 if is_approved else 0.0)

    return workflow_service.to_workflow_read(wf)


# -------------------------
# Kroger Demo UI (click-through)
# -------------------------

@router.get("/kroger", response_class=HTMLResponse)
def kroger_demo_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kroger Retail Pack — Bladnir Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; }
    header { padding: 14px 18px; border-bottom: 1px solid #e5e5e5; display:flex; gap:12px; align-items:center; }
    header b { font-size: 16px; }
    main { display:grid; grid-template-columns: 380px 1fr; height: calc(100vh - 53px); }
    .pane { padding: 14px; overflow:auto; }
    .left { border-right: 1px solid #e5e5e5; }
    .card { border: 1px solid #e5e5e5; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
    input, select, textarea { width:100%; padding:10px; border:1px solid #dcdcdc; border-radius: 8px; box-sizing:border-box; }
    textarea { min-height: 70px; }
    button { padding: 10px 12px; border: 1px solid #111; background:#111; color:#fff; border-radius: 8px; cursor:pointer; }
    button.secondary { background:#fff; color:#111; }
    button.small { padding: 8px 10px; font-size: 12px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    .muted { color:#666; font-size: 12px; }
    .badge { display:inline-block; padding: 3px 8px; border-radius: 999px; border:1px solid #ddd; font-size:12px; }
    pre { background:#0b0b0b; color:#c6ff9a; padding: 12px; border-radius: 10px; overflow:auto; }
    .sectionTitle { display:flex; justify-content:space-between; align-items:center; gap:10px; }
  </style>
</head>
<body>
<header>
  <b>Kroger Retail Pack</b>
  <span class="muted">Bladnir Tech • Click-through demo</span>
  <span class="muted" style="margin-left:auto" id="status">Loading…</span>
</header>

<main>
  <div class="pane left">
    <div class="card">
      <div class="sectionTitle">
        <b>Scenario</b>
        <span class="badge">Retail</span>
      </div>
      <div style="height:8px"></div>
      <select id="scenario"></select>
      <div style="height:10px"></div>
      <input id="store" placeholder="Store ID (e.g., 143)" />
      <div style="height:10px"></div>
      <input id="patient" placeholder="Patient Ref (optional)" />
      <div style="height:10px"></div>
      <select id="contactMethod">
        <option value="phone">phone</option>
        <option value="fax">fax</option>
        <option value="eRx">eRx</option>
        <option value="sms">sms</option>
      </select>
      <div style="height:10px"></div>
      <button onclick="startScenario()">Start Refill Case (Outbound)</button>
      <button class="secondary" style="margin-top:10px;" onclick="loadScenarios()">Reload Scenarios</button>
    </div>

    <div class="card">
      <div class="sectionTitle">
        <b>Kroger Actions</b>
        <span class="badge" id="wfBadge">No workflow</span>
      </div>
      <div class="muted" style="margin:8px 0 10px 0;">
        This simulates external prescriber response + queue movement.
      </div>

      <button class="small" onclick="prescriberApprove()">Prescriber Approves (Rx Received)</button>

      <div style="height:10px"></div>
      <div class="muted">Data Entry → Pre-Verification (runs insurance during push)</div>
      <div style="height:8px"></div>
      <textarea id="sig" placeholder="SIG"></textarea>
      <div style="height:8px"></div>
      <input id="days" type="number" placeholder="Days supply" />
      <div style="height:8px"></div>
      <input id="payer" placeholder="Payer (e.g., Express Scripts)" />
      <div style="height:8px"></div>
      <select id="insResult">
        <option value="accepted">accepted</option>
        <option value="rejected">rejected</option>
      </select>
      <div style="height:10px"></div>
      <button class="small" onclick="submitPreverify()">Push to Pre-Verification</button>

      <div style="height:10px"></div>
      <div class="muted">Pharmacist Pre-Verification</div>
      <div style="height:8px"></div>
      <div class="row">
        <button class="small" onclick="pharmDecision('approved')">Approve</button>
        <button class="small secondary" onclick="pharmDecision('rejected')">Reject</button>
      </div>
    </div>
  </div>

  <div class="pane">
  <div class="card">
    <div class="sectionTitle">
      <b>Live Workflow View</b>
      <span class="badge" id="state">—</span>
    </div>

    <div class="muted" id="meta" style="margin-top:6px;">
      No workflow selected.
    </div>

    <div style="height:14px"></div>

    <div class="row" style="gap:12px">
      <div style="flex:1" class="card">
        <b>Current Queue</b>
        <div id="queue" class="muted" style="margin-top:6px">—</div>
      </div>

      <div style="flex:1" class="card">
        <b>Insurance</b>
        <div id="insurance" class="muted" style="margin-top:6px">—</div>
      </div>

      <div style="flex:1" class="card">
        <b>Status</b>
        <div id="statusText" class="muted" style="margin-top:6px">—</div>
      </div>
    </div>

    <div style="height:14px"></div>

    <div class="row" style="gap:12px; align-items:flex-start">
      <div style="flex:1">
        <div class="sectionTitle">
          <b>Tasks</b>
          <span class="badge" id="taskCount">0</span>
        </div>
        <div id="tasks" style="margin-top:10px"></div>
      </div>

      <div style="flex:1">
        <div class="sectionTitle">
          <b>Event Timeline</b>
          <span class="badge" id="eventCount">0</span>
        </div>
        <div id="events" style="margin-top:10px"></div>
      </div>
    </div>

    <div style="height:14px"></div>

    <button class="secondary" onclick="toggleJson()">
      Advanced: Toggle JSON
    </button>

    <pre id="out"
         style="display:none;margin-top:12px;background:#0b0b0b;color:#c6ff9a;
                padding:12px;border-radius:10px;overflow:auto">{}</pre>
  </div>
</div>

</main>

<script>
  let selectedWorkflowId = null;
  let scenarioDefaults = {};

  function setStatus(msg) { document.getElementById("status").textContent = msg; }

  async function api(path, opts={}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch(e) {}
    if (!res.ok) {
      const detail = (data && (data.detail || data.message)) ? (data.detail || data.message) : res.statusText;
      throw new Error(detail);
    }
    return data;
  }

  function pretty(obj){ return JSON.stringify(obj, null, 2); }

  async function loadScenarios(){
    setStatus("Loading scenarios…");
    const sel = document.getElementById("scenario");
    sel.innerHTML = "";
    const rows = await api("/kroger/scenarios");
    rows.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.key;
      opt.textContent = r.name;
      sel.appendChild(opt);
    });
    // stash defaults for quick autofill
    scenarioDefaults = {};
    rows.forEach(r => scenarioDefaults[r.key] = r.defaults || {});
    autofillDefaults();
    setStatus("Ready");
  }

  function autofillDefaults(){
    const key = document.getElementById("scenario").value;
    const d = scenarioDefaults[key] || {};
    document.getElementById("sig").value = d.sig || "";
    document.getElementById("days").value = d.days_supply || 90;
  }

  document.getElementById("scenario").addEventListener("change", autofillDefaults);

  async function startScenario(){
    const scenario_key = document.getElementById("scenario").value;
    const store_id = document.getElementById("store").value.trim() || null;
    const patient_ref = document.getElementById("patient").value.trim() || null;
    const contact_method = document.getElementById("contactMethod").value;

    setStatus("Starting case…");
    const wf = await api("/kroger/start", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ scenario_key, store_id, patient_ref, contact_method })
    });
    selectedWorkflowId = wf.id;
    document.getElementById("wfBadge").textContent = "WF #" + wf.id;
    render(wf);
    setStatus("Ready");
  }

  async function refresh(){
    if (!selectedWorkflowId) return;
    const wf = await api("/workflows/" + selectedWorkflowId);
    render(wf);
  }

  function toggleJson(){
  const el = document.getElementById("out");
  el.style.display = (el.style.display === "none") ? "block" : "none";
}

function findLatestEvent(wf, type){
  const evs = (wf.events || []).filter(e => e.event_type === type);
  return evs.length ? evs[evs.length - 1] : null;
}

function getCurrentQueue(wf){
  const ev = findLatestEvent(wf, "queue_changed");
  if (ev && ev.payload && ev.payload.to) return ev.payload.to;

  const seed =
    findLatestEvent(wf, "refill_request_initiated") ||
    findLatestEvent(wf, "contact_event_created");

  if (seed && seed.payload && seed.payload.queue) return seed.payload.queue;

  return "—";
}

function getInsuranceStatus(wf){
  const ev = findLatestEvent(wf, "insurance_adjudicated");
  if (!ev) return "—";

  const payer = ev.payload?.payer || "payer";
  const result = ev.payload?.result || "unknown";
  return `${payer}: ${result}`;
}

function render(wf){

  // Keep JSON available for debugging (hidden by default)
  document.getElementById("out").textContent = pretty(wf);

  // Top meta info
  document.getElementById("state").textContent = wf.state;
  document.getElementById("meta").textContent =
    `#${wf.id} • ${wf.name}`;

  // Queue + Insurance cards
  document.getElementById("queue").textContent = getCurrentQueue(wf);
  document.getElementById("insurance").textContent = getInsuranceStatus(wf);
  document.getElementById("statusText").textContent = wf.state;

  // ------------------------
  // Render Tasks
  // ------------------------
  const tasks = wf.tasks || [];
  document.getElementById("taskCount").textContent = String(tasks.length);

  const tasksEl = document.getElementById("tasks");
  tasksEl.innerHTML = "";

  if (!tasks.length){
    tasksEl.innerHTML = `<div class="smallmuted">No tasks yet.</div>`;
  }

  tasks.forEach(t => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <b>${t.name}</b>
      <div class="kv">Assigned: ${t.assigned_to || "—"}</div>
      <div class="kv">State: ${t.state}</div>
    `;
    tasksEl.appendChild(div);
  });

  // ------------------------
  // Render Event Timeline
  // ------------------------
  const events = (wf.events || []).slice().reverse();
  document.getElementById("eventCount").textContent = String(events.length);

  const evEl = document.getElementById("events");
  evEl.innerHTML = "";

  if (!events.length){
    evEl.innerHTML = `<div class="smallmuted">No events yet.</div>`;
  }

  events.forEach(e => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <b>${e.event_type}</b>
      <div class="smallmuted">${JSON.stringify(e.payload || {})}</div>
    `;
    evEl.appendChild(div);
  });
}


  async function prescriberApprove(){
    if (!selectedWorkflowId) return alert("Start a case first.");
    setStatus("Simulating prescriber approval…");
    await api("/kroger/prescriber-approval", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        workflow_id: selectedWorkflowId,
        prescriber_office: "Provider Office",
        method: "eRx"
      })
    });
    await refresh();
    setStatus("Ready");
  }

  async function submitPreverify(){
    if (!selectedWorkflowId) return alert("Start a case first.");
    const sig = document.getElementById("sig").value.trim();
    const days_supply = Number(document.getElementById("days").value || 90);
    const payer = document.getElementById("payer").value.trim() || "Unknown";
    const insurance_result = document.getElementById("insResult").value;

    if (!sig) return alert("SIG is required.");

    setStatus("Submitting to pre-verification…");
    await api("/kroger/submit-preverification", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ workflow_id: selectedWorkflowId, sig, days_supply, payer, insurance_result })
    });
    await refresh();
    setStatus("Ready");
  }

  async function pharmDecision(decision){
    if (!selectedWorkflowId) return alert("Start a case first.");
    setStatus("Recording pharmacist decision…");
    await api("/kroger/pharmacist-preverify", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ workflow_id: selectedWorkflowId, decision })
    });
    await refresh();
    setStatus("Ready");
  }

  loadScenarios();
</script>

</body>
</html>
    """
