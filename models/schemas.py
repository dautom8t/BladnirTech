"""
Pydantic models and enums used throughout the Bladnir Tech API.

These models define the shape of request and response bodies and enforce
data validation at the API boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowState(str, Enum):
    ORDERED = "ordered"
    PENDING_ACCESS = "pending_access"
    ACCESS_GRANTED = "access_granted"
    DISPENSED = "dispensed"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskBase(BaseModel):
    name: str = Field(..., description="Human-readable name of the task")
    assigned_to: Optional[str] = Field(None, description="Assignee user/system identifier")


class TaskCreate(TaskBase):
    pass


class TaskRead(TaskBase):
    id: int
    workflow_id: int
    state: TaskState
    created_at: datetime
    updated_at: datetime


class EventBase(BaseModel):
    event_type: str = Field(..., description="Event type/name")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON payload")


class EventCreate(EventBase):
    pass


class EventRead(EventBase):
    id: int
    workflow_id: int
    created_at: datetime


class WorkflowBase(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")


class WorkflowCreate(WorkflowBase):
    initial_tasks: List[TaskCreate] = Field(default_factory=list)


class WorkflowRead(WorkflowBase):
    id: int
    state: WorkflowState
    created_at: datetime
    updated_at: datetime
    tasks: List[TaskRead] = Field(default_factory=list)
    events: List[EventRead] = Field(default_factory=list)


class RuleBase(BaseModel):
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    condition: str = Field(..., description="Expression to evaluate against workflow")
    action: str = Field(..., description="Expression to run when condition is true")


class RuleCreate(RuleBase):
    pass


class RuleRead(RuleBase):
    id: int
