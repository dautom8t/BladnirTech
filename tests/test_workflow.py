"""Unit tests for the workflow state machine and CRUD operations."""

from __future__ import annotations

import pytest
from models.schemas import (
    WorkflowCreate,
    TaskCreate,
    EventCreate,
    WorkflowState,
    TaskState,
)
from services.workflow import (
    create_workflow,
    get_workflow,
    list_workflows,
    add_task,
    add_event,
    update_task_state,
    to_workflow_read,
    _safe_json_loads,
)


class TestWorkflowCRUD:
    """Tests for workflow create / read / list."""

    def test_create_workflow_minimal(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF1"))
        assert wf.id is not None
        assert wf.name == "WF1"
        assert wf.state == WorkflowState.ORDERED

    def test_create_workflow_with_tasks(self, db_session):
        wf = create_workflow(
            db_session,
            WorkflowCreate(
                name="WF2",
                initial_tasks=[
                    TaskCreate(name="Task A"),
                    TaskCreate(name="Task B", assigned_to="alice"),
                ],
            ),
        )
        assert len(wf.tasks) == 2
        assert wf.tasks[0].state == TaskState.PENDING
        assert wf.tasks[1].assigned_to == "alice"

    def test_get_workflow_found(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF3"))
        fetched = get_workflow(db_session, wf.id)
        assert fetched is not None
        assert fetched.id == wf.id

    def test_get_workflow_not_found(self, db_session):
        assert get_workflow(db_session, 99999) is None

    def test_list_workflows_empty(self, db_session):
        assert list_workflows(db_session) == []

    def test_list_workflows(self, db_session):
        create_workflow(db_session, WorkflowCreate(name="A"))
        create_workflow(db_session, WorkflowCreate(name="B"))
        assert len(list_workflows(db_session)) == 2


class TestTaskOperations:
    """Tests for adding and updating tasks."""

    def test_add_task(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF"))
        task = add_task(db_session, wf.id, TaskCreate(name="New Task"))
        assert task.id is not None
        assert task.workflow_id == wf.id
        assert task.state == TaskState.PENDING

    def test_add_task_to_nonexistent_workflow(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            add_task(db_session, 99999, TaskCreate(name="orphan"))

    def test_update_task_state(self, db_session):
        wf = create_workflow(
            db_session,
            WorkflowCreate(name="WF", initial_tasks=[TaskCreate(name="T1")]),
        )
        task = wf.tasks[0]
        updated = update_task_state(db_session, wf.id, task.id, TaskState.IN_PROGRESS)
        assert updated.state == TaskState.IN_PROGRESS

    def test_update_task_state_to_completed(self, db_session):
        wf = create_workflow(
            db_session,
            WorkflowCreate(name="WF", initial_tasks=[TaskCreate(name="T1")]),
        )
        task = wf.tasks[0]
        update_task_state(db_session, wf.id, task.id, TaskState.IN_PROGRESS)
        updated = update_task_state(db_session, wf.id, task.id, TaskState.COMPLETED)
        assert updated.state == TaskState.COMPLETED

    def test_update_task_state_to_failed(self, db_session):
        wf = create_workflow(
            db_session,
            WorkflowCreate(name="WF", initial_tasks=[TaskCreate(name="T1")]),
        )
        task = wf.tasks[0]
        updated = update_task_state(db_session, wf.id, task.id, TaskState.FAILED)
        assert updated.state == TaskState.FAILED

    def test_update_nonexistent_task(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF"))
        with pytest.raises(ValueError, match="not found"):
            update_task_state(db_session, wf.id, 99999, TaskState.COMPLETED)


class TestEventOperations:
    """Tests for adding events."""

    def test_add_event(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF"))
        event = add_event(
            db_session,
            wf.id,
            EventCreate(event_type="test_event", payload={"key": "value"}),
        )
        assert event.id is not None
        assert event.event_type == "test_event"

    def test_add_event_to_nonexistent_workflow(self, db_session):
        with pytest.raises(ValueError, match="not found"):
            add_event(db_session, 99999, EventCreate(event_type="x"))

    def test_add_event_empty_payload(self, db_session):
        wf = create_workflow(db_session, WorkflowCreate(name="WF"))
        event = add_event(db_session, wf.id, EventCreate(event_type="ping"))
        assert event.payload == "{}"


class TestWorkflowConversion:
    """Tests for ORM-to-Pydantic conversion."""

    def test_to_workflow_read(self, db_session):
        wf = create_workflow(
            db_session,
            WorkflowCreate(
                name="WF",
                description="desc",
                initial_tasks=[TaskCreate(name="T1")],
            ),
        )
        add_event(db_session, wf.id, EventCreate(event_type="test", payload={"a": 1}))
        db_session.refresh(wf)

        read = to_workflow_read(wf)
        assert read.id == wf.id
        assert read.name == "WF"
        assert read.description == "desc"
        assert len(read.tasks) == 1
        assert len(read.events) == 1
        assert read.tasks[0].name == "T1"

    def test_safe_json_loads_valid(self):
        assert _safe_json_loads('{"a": 1}') == {"a": 1}

    def test_safe_json_loads_empty(self):
        assert _safe_json_loads("") == {}
        assert _safe_json_loads(None) == {}

    def test_safe_json_loads_invalid(self):
        assert _safe_json_loads("not json") == {}

    def test_safe_json_loads_non_dict(self):
        assert _safe_json_loads("[1,2,3]") == {"value": [1, 2, 3]}
