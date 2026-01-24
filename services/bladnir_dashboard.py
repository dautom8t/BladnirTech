# =============================
# API: Scenarios list (for dropdown)
# =============================

from fastapi import APIRouter, Body, Depends, HTTPException
from datetime import datetime
# other imports...

router = APIRouter(tags=["dashboard"])

DEMO_SCENARIOS = {...}
DEMO_ROWS = []
DEMO_BY_ID = {}


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

    /* If you already expanded to more queues, keep your repeat(N) columns; leaving as-is here */
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
    // Keep your current queues here. If you expanded to inbound/dispensing/etc,
    // update this object + order list accordingly.
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
    await api("/dashboard/api/auto-step", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ workflow_id: selected.id })
    });
    await refreshAll();
    setStatus("Ready");
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
    AUTH = d2.authorizations || [];
    renderBoard();

    if(selected){
      const wf = ALL.find(x => x.id === selected.id);
      if(wf) renderDetails(wf);
    }
    setStatus("Ready");
  }

  // boot
  (async () => {
    await loadScenarios();
    await refreshAll();
  })();
</script>
</body>
</html>
    """
