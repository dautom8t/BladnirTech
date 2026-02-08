"""
Workflow orchestration engine.

This module defines SQLAlchemy ORM models for workflows, tasks and events, and
provides high level functions to manipulate them. The workflow engine persists
workflows and their related tasks/events, allows creation and updates, and
integrates with the rules engine to trigger state transitions.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session, joinedload, relationship

from models import database
from models.schemas import (
    EventCreate,
    EventRead,
    TaskCreate,
    TaskRead,
    TaskState,
    WorkflowCreate,
    WorkflowRead,
    WorkflowState,
)
from services.cache import cache
from services.rules import evaluate_rules_for_workflow


class Workflow(database.Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    state = Column(SAEnum(WorkflowState), nullable=False, default=WorkflowState.ORDERED)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    tasks = relationship("Task", back_populates="workflow", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="workflow", cascade="all, delete-orphan")

    def update_timestamp(self) -> None:
        self.updated_at = datetime.utcnow()


class Task(database.Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    name = Column(String, nullable=False)
    assigned_to = Column(String, nullable=True)
    state = Column(SAEnum(TaskState), nullable=False, default=TaskState.PENDING)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    workflow = relationship("Workflow", back_populates="tasks")

    def update_timestamp(self) -> None:
        self.updated_at = datetime.utcnow()


class Event(database.Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    event_type = Column(String, nullable=False)
    payload = Column(Text, nullable=False, default="{}")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workflow = relationship("Workflow", back_populates="events")


# -------------------------
# Workflow manipulation API
# -------------------------


def create_workflow(db: Session, workflow_in: WorkflowCreate) -> Workflow:
    """Create a workflow with optional initial tasks, evaluate rules, and persist."""
    workflow = Workflow(
        name=workflow_in.name,
        description=workflow_in.description,
        state=WorkflowState.ORDERED,
    )
    db.add(workflow)
    db.flush()  # assign workflow.id before adding tasks

    if workflow_in.initial_tasks:
        for task_in in workflow_in.initial_tasks:
            db.add(
                Task(
                    workflow_id=workflow.id,
                    name=task_in.name,
                    assigned_to=task_in.assigned_to,
                    state=TaskState.PENDING,
                )
            )

    workflow.update_timestamp()

    # Run rules before final commit so everything persists in one transaction
    evaluate_rules_for_workflow(db, workflow)

    db.commit()
    db.refresh(workflow)
    cache.delete(f"workflow:{workflow.id}")
    return workflow


def get_workflow(db: Session, workflow_id: int) -> Optional[Workflow]:
    """Retrieve a workflow by ID (cache-backed in demo mode)."""
    cache_key = f"workflow:{workflow_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if wf is not None:
        cache.set(cache_key, wf, ttl=30)
    return wf


def list_workflows(db: Session) -> List[Workflow]:
    """List all workflows (eager-loads tasks and events to avoid N+1)."""
    return (
        db.query(Workflow)
        .options(joinedload(Workflow.tasks), joinedload(Workflow.events))
        .all()
    )


def add_task(db: Session, workflow_id: int, task_in: TaskCreate) -> Task:
    """Add a new task to a workflow, evaluate rules, and persist."""
    workflow = get_workflow(db, workflow_id)
    if workflow is None:
        raise ValueError(f"Workflow {workflow_id} not found")

    task = Task(
        workflow_id=workflow_id,
        name=task_in.name,
        assigned_to=task_in.assigned_to,
        state=TaskState.PENDING,
    )
    db.add(task)

    workflow.update_timestamp()

    evaluate_rules_for_workflow(db, workflow)

    db.commit()
    db.refresh(workflow)
    db.refresh(task)
    cache.delete(f"workflow:{workflow_id}")
    return task


def add_event(db: Session, workflow_id: int, event_in: EventCreate) -> Event:
    """Add an event to a workflow, evaluate rules, and persist."""
    workflow = get_workflow(db, workflow_id)
    if workflow is None:
        raise ValueError(f"Workflow {workflow_id} not found")

    event = Event(
        workflow_id=workflow_id,
        event_type=event_in.event_type,
        payload=json.dumps(event_in.payload or {}),
    )
    db.add(event)

    workflow.update_timestamp()

    evaluate_rules_for_workflow(db, workflow)

    db.commit()
    db.refresh(workflow)
    db.refresh(event)
    cache.delete(f"workflow:{workflow_id}")
    return event


def update_task_state(db: Session, workflow_id: int, task_id: int, new_state: TaskState) -> Task:
    """Update the state of a task, evaluate rules, and persist."""
    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.workflow_id == workflow_id)
        .first()
    )
    if task is None:
        raise ValueError(f"Task {task_id} not found in workflow {workflow_id}")

    task.state = new_state
    task.update_timestamp()

    workflow = task.workflow
    workflow.update_timestamp()

    evaluate_rules_for_workflow(db, workflow)

    db.commit()
    db.refresh(workflow)
    db.refresh(task)
    cache.delete(f"workflow:{workflow_id}")
    return task


# -------------------------
# ORM -> Pydantic conversion
# -------------------------


def _safe_json_loads(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return {}


def to_workflow_read(workflow: Workflow) -> WorkflowRead:
    """Convert an ORM workflow to a Pydantic response model."""
    tasks: List[TaskRead] = [
        TaskRead(
            id=t.id,
            workflow_id=t.workflow_id,
            name=t.name,
            assigned_to=t.assigned_to,
            state=t.state,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in (workflow.tasks or [])
    ]

    events: List[EventRead] = [
        EventRead(
            id=e.id,
            workflow_id=e.workflow_id,
            event_type=e.event_type,
            payload=_safe_json_loads(e.payload),
            created_at=e.created_at,
        )
        for e in (workflow.events or [])
    ]

    return WorkflowRead(
        id=workflow.id,
        name=workflow.name,
        description=workflow.description,
        state=workflow.state,
        created_at=workflow.created_at,
        updated_at=workflow.updated_at,
        tasks=tasks,
        events=events,
    )
