"""
Shared test fixtures for BladnirTech.

Provides isolated SQLite databases, FastAPI test clients, and
helper factories for creating test data.
"""

from __future__ import annotations

import os
import hashlib

# Force demo mode for all tests
os.environ["ENVIRONMENT"] = "demo"
os.environ["AUTO_SEED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./test_bladnir.db"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from models.database import Base, get_db
from src.enterprise.ame.models import Base as AMEBase


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_engine(tmp_path):
    """Create a fresh SQLite engine per test."""
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(bind=engine)
    AMEBase.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Provide an isolated database session that rolls back after each test."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False,
                           expire_on_commit=False)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(db_engine):
    """FastAPI TestClient wired to a throwaway database."""
    from main import app

    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False,
                           expire_on_commit=False)

    def _override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Auth-enabled client for testing production auth
# ---------------------------------------------------------------------------

_TEST_ADMIN_KEY = "test-admin-key-for-ci-must-be-32-chars!!"


@pytest.fixture()
def auth_client(db_engine, monkeypatch):
    """
    TestClient with authentication ENABLED.

    Sets ENVIRONMENT=production and provides a known admin key.
    The test can authenticate with headers={"X-API-Key": auth_key}.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("BLADNIR_ADMIN_KEY", _TEST_ADMIN_KEY)

    # Re-import auth module to pick up new env
    import importlib
    import enterprise.auth as auth_mod
    import config as config_mod

    # Rebuild settings for production mode
    from config import AppMode, Settings
    prod_settings = Settings()

    # Patch the auth module's key store
    key_hash = hashlib.sha256(_TEST_ADMIN_KEY.encode()).hexdigest()
    monkeypatch.setattr(auth_mod, "API_KEYS", {key_hash: ("admin", "admin")})
    monkeypatch.setattr(auth_mod, "_settings", prod_settings)

    from main import app

    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False,
                           expire_on_commit=False)

    def _override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_key():
    """The admin API key used by auth_client."""
    return _TEST_ADMIN_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_workflow(db_session):
    """Factory fixture to create a workflow in the test DB."""
    from services.workflow import Workflow, Task
    from models.schemas import WorkflowState, TaskState

    def _factory(name="Test Workflow", state=WorkflowState.ORDERED, task_names=None):
        wf = Workflow(name=name, state=state)
        db_session.add(wf)
        db_session.flush()
        if task_names:
            for tn in task_names:
                db_session.add(Task(workflow_id=wf.id, name=tn, state=TaskState.PENDING))
        db_session.commit()
        db_session.refresh(wf)
        return wf

    return _factory
