"""
Bladnir Tech — Control Tower Dashboard (TJM-style)

Adds:
- /dashboard : queue-board UI + case viewer + automation toggles
- Authorized Automation: human permission gates for auto-advancing steps
- Render-friendly: no external services required

IMPORTANT (demo):
- Automation authorizations are stored IN MEMORY (process-local).
  Good for demos. For pilots, store in DB (Postgres) with per-org scoping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from models.database import get_db
from services import workflow as workflow_service
from models.schemas import EventCreate

router = APIRouter(tags=["dashboard"])

# -----------------------------
# In-memory automation registry
# -----------------------------
# Keyed by "transition_key" (string), value: enabled bool
# For multi-tenant later: key could be (org_id, store_id, transition_key)
AUTOMATION_AUTH: Dict[str, bool] = {
    # default OFF
    "kroger.prescriber_approval_to_data_entry": False,
    "kroger.data_entry_to_preverify_insurance": False,
    "kroger.preverify_to_access_granted": False,
}

# -----------------------------
# Helpers: derive queue/status
# -----------------------------

def _latest_event_of_type(wf_read: dict, event_type: str) -> Optional[dict]:
    evs = [e for e in (wf_read.get("events") or []) if e.get("event_type") == event_type]
    return evs[-1] if evs else None

def _current_queue(wf_read: dict) -> str:
    q = _latest_event_of_type(wf_read, "queue_changed")
    if q and q.get("payload") and q["payload"].get("to"):
        return q["payload"]["to"]
    seed = _latest_event_of_type(wf_read, "refill_request_initiated") or _latest_event_of_type(wf_read, "contact_event_created")
    if seed and seed.get("payload") and seed["payload"].get("queue"):
        return seed["payload"]["queue"]
    return "unknown"

def _insurance_status(wf_read: dict) -> str:
    ins = _latest_event_of_type(wf_read, "insurance_adjudicated")
    if not ins:
        return "—"
    payload = ins.get("payload") or {}
    payer = payload.get("payer", "payer")
    result = payload.get("result", "unknown")
    return f"{payer}: {result}"

def _is_kroger_case(wf_read: dict) -> bool:
    name = (wf_read.get("name") or "").lower()
    return "kroger" in name

def _safe_name(wf_read: dict) -> str:
    return wf_read.get("name") or f"Workflow #{wf_read.get('id')}"

# -----------------------------
# API: Dashboard data
# -----------------------------

@router.get("/dashboard/api/automation")
def get_automation_state():
    return {"authorizations": AUTOMATION_AUTH}

@router.post("/dashboard/api/automation")
def set_automation_state(
    transition_key: str = Body(..., embed=True),
    enabled: bool = Body(..., embed=True),
):
    if transition_key not in AUTOMATION_AUTH:
        # allow new keys without redeploy
        AUTOMATION_AUTH[transition_key] = enabled
    else:
        AUTOMATION_AUTH[transition_key] = enabled
    return {"ok": True, "authorizations": AUTOMATION_AUTH}

@router.get("/dashboard/api/workflows")
def dashboard_list_workflows(db=Depends(get_db)):
    # Uses your workflow engine list
    wfs = workflow_service.list_workflows(db)
    wf_reads: List[dict] = [workflow_service.to_workflow_read(wf).model_dump() for wf in wfs]

    # For dashboard: show Kroger cases first, then others
    rows = []
    for wf in wf_reads:
        queue = _current_queue(wf)
        rows.append({
            "id": wf["id"],
            "name": _safe_name(wf),
            "state": wf["state"],
            "queue": queue,
            "insurance": _insurance_status(wf),
            "tasks": len(wf.get("tasks") or []),
            "events": len(wf.get("events") or []),
            "is_kroger": _is_kroger_case(wf),
            "raw": wf,  # used when clicking into a case
        })

    rows.sort(key=lambda r: (not r["is_kroger"], r["id"]))  # kroger first, then id
    return {"workflows": rows}

# -----------------------------
# Authorized Automation: auto-step
# -----------------------------
# This calls your existing Kroger pack endpoints by emitting the SAME events/tasks
# you currently do from buttons, but automatically once authorized.

def _add_event(db, workflow_id: int, event_type: str, payload: Optional[dict] = None) -> None:
    workflow_service.add_event(db, workflow_id, EventCreate(event_type=event_type, payload=payload or {}))

def _auto_step_kroger(db, wf_id: int) -> dict:
    wf = workflow_service.get_workflow(db, wf_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf_read = workflow_service.to_workflow_read(wf).model_dump()
    q = _current_queue(wf_read)

    # Step rules (simple demo logic):
    # contact_manager -> data_entry requires prescriber approval event
    # data_entry -> pre_verification includes insurance event
    # pre_verification -> access_granted requires pharmacist approval event

    if q == "contact_manager":
        if AUTOMATION_AUTH.get("kroger.prescriber_approval_to_data_entry"):
            _add_event(db, wf_id, "prescriber_refill_approved", {"method": "eRx"})
            _add_event(db, wf_id, "authorization_received", {"rx_status": "received"})
            _add_event(db, wf_id, "queue_changed", {"from": "contact_manager", "to": "data_entry"})
            _add_event(db, wf_id, "contact_manager_removed", {"reason": "automation"})
        else:
            return {"did": "none", "reason": "not_authorized", "next": "prescriber_approval_to_data_entry"}

    elif q == "data_entry":
        if AUTOMATION_AUTH.get("kroger.data_entry_to_preverify_insurance"):
            _add_event(db, wf_id, "data_entry_completed", {"sig": "Auto SIG", "days_supply": 90})
            _add_event(db, wf_id, "insurance_adjudicated", {"payer": "AutoPayer", "result": "accepted"})
            _add_event(db, wf_id, "queue_changed", {"from": "data_entry", "to": "pre_verification"})
        else:
            return {"did": "none", "reason": "not_authorized", "next": "data_entry_to_preverify_insurance"}

    elif q == "pre_verification":
        if AUTOMATION_AUTH.get("kroger.preverify_to_access_granted"):
            _add_event(db, wf_id, "pre_verification_reviewed", {"decision": "approved", "notes": "auto"})
            # Optional: reflect workflow state in ORM (keeps your state machine feel)
            wf2 = workflow_service.get_workflow(db, wf_id)
            if wf2:
                # ACCESS_GRANTED is in your WorkflowState enum
                from models.schemas import WorkflowState
                wf2.state = WorkflowState.ACCESS_GRANTED
                wf2.update_timestamp()
                db.commit()
        else:
            return {"did": "none", "reason": "not_authorized", "next": "preverify_to_access_granted"}

    else:
        return {"did": "none", "reason": f"queue_{q}_no_autostep"}

    wf = workflow_service.get_workflow(db, wf_id)
    return {"did": "ok", "workflow": workflow_service.to_workflow_read(wf).model_dump()}

@router.post("/dashboard/api/auto-step")
def dashboard_auto_step(workflow_id: int = Body(..., embed=True), db=Depends(get_db)):
    return _auto_step_kroger(db, workflow_id)

# -----------------------------
# UI: /dashboard (TJM-style)
# -----------------------------

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
    .cols{display:grid;grid-template-columns: repeat(4, 1fr); gap:12px;}
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
  <span class="muted" id="status" style="margin-left:auto">Loading…</span>
</header>

<div class="wrap">
  <div class="left">
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <b style="font-size:14px">Queues</b>
          <div class="muted">Click a case card to inspect • Use auto-step after authorization</div>
        </div>
        <button class="small" onclick="refreshAll()">Refresh</button>
      </div>
      <div style="height:10px"></div>
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
          <button class="small" onclick="toggleJson()">Toggle JSON</button>
        </div>
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

  function setStatus(t){ document.getElementById('status').textContent = t; }

  async function api(path, opts={}){
    const res = await fetch(path, opts);
    let data = null;
    try{ data = await res.json(); }catch(e){}
    if(!res.ok){
      const msg = (data && (data.detail || data.message)) ? (data.detail || data.message) : res.statusText;
      throw new Error(msg);
    }
    return data;
  }

  function toggleJson(){
    const el = document.getElementById("json");
    el.style.display = (el.style.display === "none") ? "block" : "none";
  }

  function groupByQueue(rows){
    const cols = { contact_manager:[], data_entry:[], pre_verification:[], rejection_resolution:[] };
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
      ["contact_manager","Contact Manager"],
      ["data_entry","Data Entry"],
      ["pre_verification","Pre-Verification"],
      ["rejection_resolution","Rejections"]
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
    const res = await api("/dashboard/api/auto-step", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ workflow_id: selected.id })
    });
    await refreshAll();
    setStatus("Ready");
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
    setStatus("Ready");
  }

  refreshAll();
</script>
</body>
</html>
    """
