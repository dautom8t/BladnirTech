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
from fastapi.responses import HTMLResponse, RedirectResponse

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
# /kroger redirects to the PactGate™ dashboard
# -------------------------

@router.get("/kroger")
def kroger_redirect():
    return RedirectResponse(url="/dashboard", status_code=302)
