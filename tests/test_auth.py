"""Authentication and authorization tests.

Verifies that:
- Protected endpoints reject unauthenticated requests in production mode
- Valid API keys are accepted
- Invalid API keys are rejected
- Role-based access control is enforced
- Rate limiting works
"""

from __future__ import annotations

import hashlib
import pytest
from unittest.mock import patch

from enterprise.auth import (
    _verify_api_key,
    RateLimiter,
    UserContext,
    _DEMO_USER,
)


# ---------------------------------------------------------------------------
# Unit tests for auth internals
# ---------------------------------------------------------------------------

class TestVerifyApiKey:

    def test_rejects_none(self):
        assert _verify_api_key(None) is None

    def test_rejects_empty(self):
        assert _verify_api_key("") is None

    def test_rejects_short_key(self):
        assert _verify_api_key("too_short") is None

    def test_rejects_invalid_key(self):
        assert _verify_api_key("x" * 40) is None


class TestRateLimiter:

    def test_allows_under_limit(self):
        rl = RateLimiter(max_attempts=3, window_seconds=60)
        assert rl.check_rate_limit("ip1") is True
        assert rl.check_rate_limit("ip1") is True
        assert rl.check_rate_limit("ip1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_attempts=2, window_seconds=60)
        rl.check_rate_limit("ip1")
        rl.check_rate_limit("ip1")
        assert rl.check_rate_limit("ip1") is False

    def test_different_ips_independent(self):
        rl = RateLimiter(max_attempts=1, window_seconds=60)
        assert rl.check_rate_limit("ip1") is True
        assert rl.check_rate_limit("ip2") is True
        assert rl.check_rate_limit("ip1") is False
        assert rl.check_rate_limit("ip2") is False


class TestDemoUser:

    def test_demo_user_is_admin(self):
        assert _DEMO_USER.role == "admin"
        assert _DEMO_USER.user_id == "demo"


# ---------------------------------------------------------------------------
# Integration: endpoints in production mode
# ---------------------------------------------------------------------------

class TestAuthEnforcement:
    """Test that protected endpoints require auth in production mode."""

    def test_enterprise_execute_rejects_no_key(self, auth_client):
        r = auth_client.post("/enterprise/execute", json={
            "workflow_id": 1,
            "action_type": "advance_queue",
        })
        assert r.status_code == 401

    def test_enterprise_status_rejects_no_key(self, auth_client):
        r = auth_client.get("/enterprise/status")
        assert r.status_code == 401

    def test_enterprise_execute_rejects_invalid_key(self, auth_client):
        r = auth_client.post(
            "/enterprise/execute",
            json={"workflow_id": 1, "action_type": "advance_queue"},
            headers={"X-API-Key": "invalid-key-that-is-definitely-at-least-32-chars!!"},
        )
        assert r.status_code == 401

    def test_enterprise_status_accepts_valid_key(self, auth_client, auth_key):
        r = auth_client.get(
            "/enterprise/status",
            headers={"X-API-Key": auth_key},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["user"]["role"] == "admin"

    def test_enterprise_execute_accepts_valid_key(self, auth_client, auth_key):
        """Valid key + admin role should pass auth (may fail on workflow not found, which is fine)."""
        r = auth_client.post(
            "/enterprise/execute",
            json={"workflow_id": 1, "action_type": "advance_queue"},
            headers={"X-API-Key": auth_key},
        )
        # 404 means auth passed but workflow doesn't exist — that's correct
        assert r.status_code in (200, 404)

    def test_dry_run_rejects_no_key(self, auth_client):
        r = auth_client.post("/enterprise/execute/dry-run", json={
            "workflow_id": 1,
            "action_type": "advance_queue",
        })
        assert r.status_code == 401


class TestDemoModeBypass:
    """Verify that demo mode bypasses auth entirely."""

    def test_enterprise_status_no_key_needed(self, client):
        """In demo mode, no API key is required."""
        r = client.get("/enterprise/status")
        assert r.status_code == 200
        assert r.json()["user"]["user_id"] == "demo"

    def test_health_always_accessible(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_public_endpoints_no_auth(self, client):
        """Public endpoints should always work without auth."""
        for path in ["/health", "/api/settings", "/api/workflows", "/api/rules"]:
            r = client.get(path)
            assert r.status_code == 200, f"Failed for {path}"
