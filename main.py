"""
FastAPI application exposing the Bladnir Tech API.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import Body, Depends, FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from config import settings
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
from enterprise.auth import require_auth, require_role, UserContext

from enterprise.execute import router as enterprise_router
from src.enterprise.ame import router as ame_router
from services.kroger_retail_pack import router as kroger_router
from services.demo_hub import router as demo_router
from services.bladnir_dashboard import router as dashboard_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-create tables in demo and development modes
if settings.auto_create_tables:
    try:
        Base.metadata.create_all(bind=engine)
        # AME models use their own declarative Base
        from src.enterprise.ame.models import Base as AMEBase
        AMEBase.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")

app = FastAPI(title="Bladnir Tech - Control Tower")

# Include routers
app.include_router(kroger_router)
app.include_router(demo_router)
app.include_router(dashboard_router)
app.include_router(enterprise_router)
app.include_router(ame_router)

# Environment-based CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict:
    from services.cache import cache
    from services.storage import storage
    return {
        "status": "ok",
        "mode": settings.mode.value,
        "cache": cache.stats(),
        "storage": storage.info(),
    }

@app.get("/api/settings")
def api_settings() -> dict:
    """Expose non-sensitive runtime settings so the UI can adapt."""
    return {
        "mode": settings.mode.value,
        "auth_enabled": settings.auth_enabled,
        "cache_backend": settings.cache_backend,
        "storage_backend": settings.storage_backend,
        "auto_seed": settings.auto_seed,
    }

@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")

@app.get("/demo")
def demo_ui() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")

# =============================
# Workflow API
# =============================

@app.post("/api/workflows", response_model=WorkflowRead, status_code=status.HTTP_201_CREATED)
def api_create_workflow(workflow_in: WorkflowCreate, user: UserContext = Depends(require_auth), db: Session = Depends(get_db)):
    wf = workflow_service.create_workflow(db, workflow_in)
    return workflow_service.to_workflow_read(wf)

@app.get("/api/workflows", response_model=List[WorkflowRead])
def api_list_workflows(user: UserContext = Depends(require_auth), db: Session = Depends(get_db)):
    wfs = workflow_service.list_workflows(db)
    return [workflow_service.to_workflow_read(wf) for wf in wfs]

@app.get("/api/workflows/{workflow_id}", response_model=WorkflowRead)
def api_get_workflow(workflow_id: int = Path(..., gt=0), user: UserContext = Depends(require_auth), db: Session = Depends(get_db)):
    wf = workflow_service.get_workflow(db, workflow_id)
    if wf is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    return workflow_service.to_workflow_read(wf)

# =============================
# Task API
# =============================

@app.post("/api/workflows/{workflow_id}/tasks", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def api_add_task(
    workflow_id: int = Path(..., gt=0),
    task_in: TaskCreate = Body(...),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        task = workflow_service.add_task(db, workflow_id, task_in)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    
    # FIXED: Use helper function for consistency
    return TaskRead(
        id=task.id,
        workflow_id=task.workflow_id,
        name=task.name,
        assigned_to=task.assigned_to,
        state=task.state,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )

@app.patch("/api/workflows/{workflow_id}/tasks/{task_id}/state", response_model=TaskRead)
def api_update_task_state(
    workflow_id: int = Path(..., gt=0),
    task_id: int = Path(..., gt=0),
    new_state: TaskState = Body(...),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        task = workflow_service.update_task_state(db, workflow_id, task_id, new_state)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task or workflow not found")
    
    return TaskRead(
        id=task.id,
        workflow_id=task.workflow_id,
        name=task.name,
        assigned_to=task.assigned_to,
        state=task.state,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )

# =============================
# Event API
# =============================

@app.post("/api/workflows/{workflow_id}/events", response_model=EventRead, status_code=status.HTTP_201_CREATED)
def api_add_event(
    workflow_id: int = Path(..., gt=0),
    event_in: EventCreate = Body(...),
    user: UserContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        event = workflow_service.add_event(db, workflow_id, event_in)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow not found")
    
    # FIXED: Use event.payload instead of event_in.payload
    return EventRead(
        id=event.id,
        workflow_id=event.workflow_id,
        event_type=event.event_type,
        payload=event.payload,
        created_at=event.created_at,
    )

# =============================
# Rules API
# =============================

@app.post("/api/rules", response_model=RuleRead, status_code=status.HTTP_201_CREATED)
def api_create_rule(rule_in: RuleCreate, user: UserContext = Depends(require_role("admin")), db: Session = Depends(get_db)):
    rule = rules_service.create_rule(db, rule_in)
    return RuleRead(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        condition=rule.condition,
        action=rule.action,
    )

@app.get("/api/rules", response_model=List[RuleRead])
def api_list_rules(user: UserContext = Depends(require_auth), db: Session = Depends(get_db)):
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


# =============================
# Startup: auto-seed in demo mode
# =============================

@app.on_event("startup")
async def _on_startup() -> None:
    logger.info(f"BladnirTech starting in {settings.mode.value} mode")
    if settings.is_demo and settings.auto_seed:
        from services.bladnir_dashboard import seed_demo_cases, DEMO_ROWS
        if not DEMO_ROWS:
            seed_demo_cases(scenario_id="happy_path", seed_all=True)
            logger.info(f"Auto-seeded {len(DEMO_ROWS)} demo cases on startup")
