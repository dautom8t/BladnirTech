"""
FastAPI application exposing the PharmAI middleware API.

This module wires together the workflow engine, rules engine and integration
stubs into a RESTful API.  It demonstrates how you might build a middleware
service that coordinates medication workflows across disparate systems.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware

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
from services import workflow as workflow_service
from services import rules as rules_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create all tables at startup.  In production you may use alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="PharmAI Middleware API", version="0.1.0")

# Allow cross‑origin requests for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/workflows", response_model=WorkflowRead, status_code=status.HTTP_201_CREATED)
def api_create_workflow(workflow_in: WorkflowCreate, db=Depends(get_db)):
    """Create a new workflow along with optional initial tasks."""
    wf = workflow_service.create_workflow(db, workflow_in)
    return workflow_service.to_workflow_read(wf)


@app.get("/workflows", response_model=List[WorkflowRead])
def api_list_workflows(db=Depends(get_db)):
    """List all workflows."""
    wfs = workflow_service.list_workflows(db)
    return [workflow_service.to_workflow_read(wf) for wf in wfs]


@app.get("/workflows/{workflow_id}", response_model=WorkflowRead)
def api_get_workflow(workflow_id: int = Path(..., gt=0), db=Depends(get_db)):
    """Retrieve a workflow by its ID."""
    wf = workflow_service.get_workflow(db, workflow_id)
    if wf is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    return workflow_service.to_workflow_read(wf)


@app.post("/workflows/{workflow_id}/tasks", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def api_add_task(
    workflow_id: int = Path(..., gt=0),
    task_in: TaskCreate = None,
    db=Depends(get_db),
):
    """Add a task to a given workflow."""
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
    """Update the state of a task (e.g. mark it completed)."""
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
def api_add_event(
    workflow_id: int = Path(..., gt=0),
    event_in: EventCreate = None,
    db=Depends(get_db),
):
    """Add a generic event to a workflow (e.g. note, status update)."""
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
    """Create a new rule in the system."""
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
    """List all rules."""
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
