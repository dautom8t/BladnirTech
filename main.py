"""
FastAPI application exposing the Bladnir Tech API.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse


from models.database import Base, engine, get_db
from models.schemas import (
    EventCreate,
    EventRead,
    RuleCreate,
    RuleRead,
    TaskCreate,
    TaskRead,
    TaskState,
    WorkflowCreate,
    WorkflowRead,
)
from services import rules as rules_service
from services import workflow as workflow_service

from enterprise.execute import router as enterprise_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tables (dev-only; for prod use Alembic migrations)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bladnir Tech- Control Tower")

from services.kroger_retail_pack import router as kroger_router
app.include_router(kroger_router)

from services.demo_hub import router as demo_router
app.include_router(demo_router)

from services.bladnir_dashboard import router as dashboard_router
app.include_router(dashboard_router)

app.include_router(enterprise_router)



# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.get("/api/workflows")
def dashboard_list_workflows(db=Depends(get_db)):
    wfs = workflow_service.list_workflows(db)
    items = [workflow_service.to_workflow_read(wf) for wf in wfs]
    return {"workflows": items}

@app.get("/api/automation")
def dashboard_get_automation(db=Depends(get_db)):
    # Minimal stub so the dashboard stops 404ing.
    # You can wire this to DB later.
    return {"authorizations": {}}
    
@app.get("/")
def home():
    return RedirectResponse(url="/dashboard")


@app.get("/demo")
def demo_ui():
    return RedirectResponse(url="/dashboard")


@app.post("/workflows", response_model=WorkflowRead, status_code=status.HTTP_201_CREATED)
def api_create_workflow(workflow_in: WorkflowCreate, db=Depends(get_db)):
    wf = workflow_service.create_workflow(db, workflow_in)
    return workflow_service.to_workflow_read(wf)

@app.get("/workflows", response_model=List[WorkflowRead])
def api_list_workflows(db=Depends(get_db)):
    wfs = workflow_service.list_workflows(db)
    return [workflow_service.to_workflow_read(wf) for wf in wfs]

@app.get("/workflows/{workflow_id}", response_model=WorkflowRead)
def api_get_workflow(workflow_id: int = Path(..., gt=0), db=Depends(get_db)):
    wf = workflow_service.get_workflow(db, workflow_id)
    if wf is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    return workflow_service.to_workflow_read(wf)

@app.post("/workflows/{workflow_id}/tasks", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def api_add_task(workflow_id: int = Path(..., gt=0), task_in: TaskCreate = ... , db=Depends(get_db)):
    try:
        task = workflow_service.add_task(db, workflow_id, task_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return TaskRead(
        id=task.id,
        workflow_id=task.workflow_id,
        name=task.name,
        assigned_to=task.assigned_to,
        state=task.state,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )

@app.post("/workflows/{workflow_id}/tasks/{task_id}/state", response_model=TaskRead)
def api_update_task_state(
    workflow_id: int = Path(..., gt=0),
    task_id: int = Path(..., gt=0),
    new_state: TaskState = TaskState.COMPLETED,
    db=Depends(get_db),
):
    try:
        task = workflow_service.update_task_state(db, workflow_id, task_id, new_state)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return TaskRead(
        id=task.id,
        workflow_id=task.workflow_id,
        name=task.name,
        assigned_to=task.assigned_to,
        state=task.state,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )

@app.post("/workflows/{workflow_id}/events", response_model=EventRead, status_code=status.HTTP_201_CREATED)
def api_add_event(workflow_id: int = Path(..., gt=0), event_in: EventCreate = ..., db=Depends(get_db)):
    try:
        event = workflow_service.add_event(db, workflow_id, event_in)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return EventRead(
        id=event.id,
        workflow_id=event.workflow_id,
        event_type=event.event_type,
        payload=event_in.payload,
        created_at=event.created_at,
    )

@app.post("/rules", response_model=RuleRead, status_code=status.HTTP_201_CREATED)
def api_create_rule(rule_in: RuleCreate, db=Depends(get_db)):
    rule = rules_service.create_rule(db, rule_in)
    return RuleRead(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        condition=rule.condition,
        action=rule.action,
    )

@app.get("/rules", response_model=List[RuleRead])
def api_list_rules(db=Depends(get_db)):
    rules = rules_service.list_rules(db)
    return [
        RuleRead(
            id=r.id,
            name=r.name,
            description=r.description,
            condition=r.condition,
            action=r.action,
        )
        for r in rules
    ]
