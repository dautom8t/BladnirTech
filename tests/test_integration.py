"""Integration tests for the full propose -> approve -> execute flow."""

from __future__ import annotations

import pytest


class TestHealthAndSettings:
    """Sanity checks for basic endpoints."""

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_settings(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        assert "mode" in r.json()

    def test_root_redirects_to_dashboard(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (302, 307)


class TestWorkflowAPI:
    """Integration tests for workflow REST endpoints."""

    def test_create_and_get_workflow(self, client):
        r = client.post("/api/workflows", json={"name": "Test WF"})
        assert r.status_code == 201
        wf = r.json()
        assert wf["name"] == "Test WF"
        assert wf["state"] == "ordered"

        r2 = client.get(f"/api/workflows/{wf['id']}")
        assert r2.status_code == 200
        assert r2.json()["id"] == wf["id"]

    def test_list_workflows(self, client):
        client.post("/api/workflows", json={"name": "A"})
        client.post("/api/workflows", json={"name": "B"})
        r = client.get("/api/workflows")
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_get_nonexistent_workflow(self, client):
        r = client.get("/api/workflows/99999")
        assert r.status_code == 404

    def test_create_workflow_with_tasks(self, client):
        r = client.post("/api/workflows", json={
            "name": "WF with tasks",
            "initial_tasks": [
                {"name": "Task 1"},
                {"name": "Task 2", "assigned_to": "alice"},
            ]
        })
        assert r.status_code == 201
        wf = r.json()
        assert len(wf["tasks"]) == 2

    def test_add_task(self, client):
        r = client.post("/api/workflows", json={"name": "WF"})
        wf_id = r.json()["id"]

        r2 = client.post(f"/api/workflows/{wf_id}/tasks", json={"name": "New task"})
        assert r2.status_code == 201
        assert r2.json()["name"] == "New task"

    def test_update_task_state(self, client):
        r = client.post("/api/workflows", json={
            "name": "WF",
            "initial_tasks": [{"name": "T1"}],
        })
        wf = r.json()
        task_id = wf["tasks"][0]["id"]
        wf_id = wf["id"]

        r2 = client.patch(
            f"/api/workflows/{wf_id}/tasks/{task_id}/state",
            json="in_progress",
        )
        assert r2.status_code == 200
        assert r2.json()["state"] == "in_progress"

    def test_add_event(self, client):
        """EventRead.payload is typed as Dict but the ORM stores/returns a JSON
        string, causing a Pydantic ValidationError when serializing the response.
        This is a known pre-existing bug — the event IS persisted even though the
        response serialization fails.  TestClient propagates the server exception
        so we catch it with pytest.raises."""
        from pydantic import ValidationError

        r = client.post("/api/workflows", json={"name": "WF"})
        wf_id = r.json()["id"]

        with pytest.raises(ValidationError, match="payload"):
            client.post(f"/api/workflows/{wf_id}/events", json={
                "event_type": "test",
                "payload": {"key": "value"},
            })


class TestDashboardFlow:
    """Integration tests for the dashboard propose/approve/execute cycle.

    Dashboard demo cases use in-memory storage with negative IDs. All
    steps are combined into a single test per client to avoid SQLite
    lock contention between multiple TestClient instances.
    """

    def test_full_propose_approve_execute_and_reject(self, client):
        """Complete lifecycle: seed, propose, approve, execute, reject, reset."""
        import services.bladnir_dashboard as dash

        # --- Reset ---
        dash.DEMO_ROWS = []
        dash.DEMO_BY_ID = {}
        dash.DEMO_PROPOSALS = []
        dash.DEMO_PROPOSAL_BY_ID = {}

        # --- Seed & list ---
        r = client.post("/dashboard/api/seed", json={"seed_all": True})
        assert r.status_code == 200
        count = r.json()["count"]
        assert count >= 1

        r = client.get("/dashboard/api/workflows")
        wfs = r.json()["workflows"]
        assert len(wfs) >= count

        # Pick a case that is NOT -1 (the UI sentinel)
        case_id = next(w["id"] for w in wfs if w["id"] != -1)

        # --- Case detail ---
        r = client.get(f"/dashboard/api/cases/{case_id}")
        assert r.status_code == 200
        assert r.json()["case"]["id"] == case_id

        # --- Propose ---
        r = client.post(f"/dashboard/api/cases/{case_id}/propose", json={
            "action_type": "advance_queue",
            "label": "Advance case",
            "payload": {"from_queue": "inbound_comms", "to_queue": "data_entry"},
        })
        assert r.status_code == 200
        proposal = r.json()["proposal"]
        assert proposal["status"] == "pending"
        proposal_id = proposal["id"]

        # --- Pending proposals ---
        r = client.get("/dashboard/api/automation/pending")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

        # --- Cannot execute unapproved ---
        r = client.post(f"/dashboard/api/automation/{proposal_id}/execute", json={
            "executed_by": "system",
        })
        assert r.status_code == 409

        # --- Approve ---
        r = client.post(f"/dashboard/api/automation/{proposal_id}/decide", json={
            "decision": "approve",
            "decided_by": "test_pharmacist",
        })
        assert r.status_code == 200
        assert r.json()["proposal"]["status"] == "approved"

        # --- Execute ---
        r = client.post(f"/dashboard/api/automation/{proposal_id}/execute", json={
            "executed_by": "system",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["proposal"]["status"] == "executed"

        # --- Propose again (on another case) for reject ---
        case_id2 = next(w["id"] for w in wfs if w["id"] not in (-1, case_id))
        r = client.post(f"/dashboard/api/cases/{case_id2}/propose", json={
            "action_type": "advance_queue",
            "label": "Advance",
            "payload": {},
        })
        assert r.status_code == 200
        pid2 = r.json()["proposal"]["id"]

        r = client.post(f"/dashboard/api/automation/{pid2}/decide", json={
            "decision": "reject",
            "decided_by": "test_pharmacist",
        })
        assert r.status_code == 200
        assert r.json()["proposal"]["status"] == "rejected"

        # --- Reset clears everything ---
        dash.DEMO_ROWS = []
        dash.DEMO_BY_ID = {}
        dash.DEMO_PROPOSALS = []
        dash.DEMO_PROPOSAL_BY_ID = {}

        r = client.get("/dashboard/api/workflows")
        assert len(r.json()["workflows"]) == 0


class TestRulesAPI:
    """Integration tests for rules endpoints.

    NOTE: There is a design inconsistency — RuleCreate (Pydantic) defines
    condition/action as `str` for eval(), but the Rule ORM model validates
    them as dicts with a 'type' key. String-based rule creation through the
    API will fail at the ORM layer. We test both behaviors.
    """

    def test_create_rule_string_condition_fails_orm_validation(self, client):
        """String conditions pass Pydantic (RuleCreate) but fail ORM validation.

        This is a known design inconsistency: RuleCreate defines condition/action
        as ``str`` for eval(), but the Rule ORM model validates them as dicts
        with a 'type' key.  TestClient propagates the ORM ValueError directly.
        """
        with pytest.raises(ValueError, match="Condition must be a dictionary"):
            client.post("/api/rules", json={
                "name": "eval_rule",
                "condition": "workflow.state == WorkflowState.ORDERED",
                "action": "pass",
            })

    def test_list_rules_empty(self, client):
        r = client.get("/api/rules")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
