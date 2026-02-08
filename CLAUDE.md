# CLAUDE.md — Bladnir Tech PactGate™

## Project Overview

BladnirTech is a **workflow orchestration middleware platform** built with FastAPI. It provides **PactGate™**, a governance-first platform for managing automated workflows with human governance gates, AME™ (Adaptive Model Evolution) trust stages, an ML-driven adaptive trust system, a rules engine, and audit-first architecture. The primary demo scenario targets Kroger retail pharmacy prescription refill workflows.

## Quick Start

```bash
# Install dependencies:
pip install -r requirements.txt

# Run in demo mode (zero config, just works):
uvicorn main:app --reload

# Verify:
curl http://127.0.0.1:8000/health
# → {"status":"ok","mode":"demo",...}

# Open dashboard:
open http://127.0.0.1:8000/dashboard
```

Demo mode auto-creates tables and seeds sample data on startup. No environment variables required.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI >=0.110 (ASGI)
- **Server:** Uvicorn 0.29 (pinned), h11 0.14.0 (pinned)
- **ORM:** SQLAlchemy 2.0+ (2.0 API via `future=True`)
- **Validation:** Pydantic 2.6+
- **Database:** SQLite (demo/dev), PostgreSQL or MySQL (production)
- **ML:** scikit-learn 1.4+ (GradientBoosting, IsolationForest, CalibratedClassifier)
- **Model Persistence:** joblib 1.3+
- **Numerical:** NumPy 1.26+

## Unified Codebase, Different Environments

The same codebase runs in **demo**, **development**, and **production** modes controlled by the `ENVIRONMENT` variable. Each mode swaps backend implementations via abstraction layers:

| Component | Demo (default) | Development | Production |
|---|---|---|---|
| **Database** | SQLite | SQLite | PostgreSQL / MySQL |
| **Cache** | In-memory dict (TTL) | In-memory dict (TTL) | Redis |
| **Storage** | Local filesystem | Local filesystem | S3 / Cloud |
| **Auth** | Bypassed (auto-admin) | API key + RBAC | API key + RBAC |
| **Data seeding** | Auto-seeds on startup | Manual | Manual |
| **Table creation** | Automatic | Automatic | Manual migration |
| **SQL logging** | Off | Enabled (`echo=True`) | Off |

### Running Each Mode

```bash
# Demo (zero config, just works):
uvicorn main:app --reload

# Development (requires BLADNIR_ADMIN_KEY):
ENVIRONMENT=development BLADNIR_ADMIN_KEY=<32+ chars> uvicorn main:app --reload

# Production:
ENVIRONMENT=production DATABASE_URL=postgresql://... BLADNIR_ADMIN_KEY=<32+ chars> uvicorn main:app
```

## Repository Structure

```
BladnirTech/
├── main.py                             # FastAPI app entry point, core API routes, startup hooks
├── config.py                           # Centralized settings (AppMode, env-var parsing, frozen dataclass)
├── requirements.txt                    # Python dependencies (8 packages)
├── README.md                           # Project readme
├── CLAUDE.md                           # This file
├── .gitignore                          # Ignores __pycache__, *.db, logs/, data/, .env, IDE dirs
├── docs/
│   └── AME_ML_SPEC.md                 # ML design spec for AME trust models
├── enterprise/                         # Enterprise features
│   ├── __init__.py
│   ├── auth.py                         # API key auth, RBAC, rate limiting, demo bypass (~316 lines)
│   ├── governance.py                   # Governance gates (proposal-approval flows, JSON persistence, ~446 lines)
│   ├── audit.py                        # Audit logging to JSONL files (thread-safe, auto-rotating, ~248 lines)
│   └── execute.py                      # Governed execution router (/enterprise/*, ~380 lines)
├── models/                             # Data layer
│   ├── __init__.py
│   ├── database.py                     # SQLAlchemy engine, session, multi-DB config (~288 lines)
│   ├── schemas.py                      # Pydantic request/response models + state enums (~95 lines)
│   └── rules.py                        # Rule ORM model (condition/action pairs, ~127 lines)
├── services/                           # Business logic
│   ├── __init__.py
│   ├── workflow.py                     # Workflow/Task/Event ORM models + CRUD (cache-backed reads, ~259 lines)
│   ├── rules.py                        # Rules engine (evaluation, CRUD, ~60 lines)
│   ├── cache.py                        # Cache abstraction (MemoryCache / RedisCache, ~180 lines)
│   ├── storage.py                      # Storage abstraction (LocalStorage / S3Storage, ~177 lines)
│   ├── integration.py                  # External system integration stubs (EHR, pharmacy, payer, ~40 lines)
│   ├── kroger_retail_pack.py           # Kroger pharmacy demo scenario (/kroger/*, ~377 lines)
│   ├── bladnir_dashboard.py            # Dashboard UI + API (/dashboard/*) — largest file (~2739 lines)
│   └── demo_hub.py                     # Demo hub landing page (/demo, ~85 lines)
└── src/
    └── enterprise/
        └── ame/                        # Adaptive Model Evolution — ML-driven trust system
            ├── __init__.py             # Exports AME router
            ├── models.py               # AMETrustScope, AMEEvent, AMEExecution ORM models (~364 lines)
            ├── service.py              # Trust computation, stage management, execution tracking (~828 lines)
            ├── router.py               # REST API endpoints (/ame/*, ~945 lines)
            └── ml/                     # Machine learning layer
                ├── __init__.py         # Model factory functions
                ├── features.py         # Feature engineering for all 3 ML models (~600 lines)
                ├── decision.py         # Approval probability predictor (GradientBoosting)
                ├── outcome.py          # Success probability predictor (CalibratedClassifier)
                ├── anomaly.py          # Anomaly detector (IsolationForest)
                ├── store.py            # Model persistence (joblib, versioned)
                └── trainer.py          # Training orchestration + retrain triggers
```

**Total: ~31 Python files, ~9,900+ lines of code.**

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `demo` | `demo`, `development`, or `production` (also accepts `prod`) |
| `DATABASE_URL` | `sqlite:///./BladnirTech.db` | Database connection string |
| `BLADNIR_ADMIN_KEY` | (built-in for demo) | Admin API key (min 32 chars, required in dev/prod) |
| `BLADNIR_KEY_<NAME>` | (optional) | Service account keys, format `key:role` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection (when `CACHE_BACKEND=redis`) |
| `CACHE_TTL` | `300` | Default cache TTL in seconds |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `STORAGE_LOCAL_DIR` | `./data/files` | Local file storage root |
| `S3_BUCKET` | (none) | S3 bucket name (when `STORAGE_BACKEND=s3`) |
| `S3_REGION` | `us-east-1` | AWS region for S3 |
| `AUTO_SEED` | `true` | Auto-seed demo data on startup (demo mode) |
| `DB_POOL_SIZE` | `5` | Connection pool size (PostgreSQL/MySQL) |
| `DB_MAX_OVERFLOW` | `10` | Max pool overflow connections |
| `DB_POOL_TIMEOUT` | `30` | Pool timeout in seconds |
| `DB_POOL_RECYCLE` | `3600` | Connection recycle time in seconds |
| `AUDIT_LOG_DIR` | `./logs/audit` | Audit log directory |
| `AUDIT_MAX_SIZE_MB` | `100` | Audit log rotation threshold |

## Key API Endpoints

### Core Workflow API (`main.py`)

- `GET /` — Redirects to `/dashboard`
- `GET /demo` — Redirects to `/dashboard`
- `GET /health` — Health check (includes cache stats, storage info, mode)
- `GET /api/settings` — Non-sensitive runtime settings for UI adaptation
- `POST /api/workflows` — Create workflow
- `GET /api/workflows` — List workflows
- `GET /api/workflows/{id}` — Get workflow by ID
- `POST /api/workflows/{id}/tasks` — Add task
- `PATCH /api/workflows/{id}/tasks/{tid}/state` — Update task state
- `POST /api/workflows/{id}/events` — Add event
- `POST /api/rules` — Create rule
- `GET /api/rules` — List rules

### Enterprise Execution (`/enterprise/*`)

- `POST /enterprise/execute` — Execute governed action (admin role required)
- `GET /enterprise/status` — Enterprise status and available actions
- `POST /enterprise/execute/dry-run` — Dry-run validation

### Dashboard (`/dashboard/*`)

- `GET /dashboard` — Interactive dashboard UI (HTML + embedded JS, mobile-responsive)
- `GET /dashboard/api/automation` — Get governance gate statuses
- `POST /dashboard/api/automation` — Authorize/revoke governance gates
- `GET /dashboard/api/automation/pending` — List pending proposals
- `GET /dashboard/api/activity` — Get activity log
- `GET /dashboard/api/cases/{case_id}` — Get case detail with proposals and trust info
- `POST /dashboard/api/cases/{case_id}/propose` — Propose automated action
- `POST /dashboard/api/automation/{proposal_id}/decide` — Approve/reject proposal
- `POST /dashboard/api/automation/{proposal_id}/execute` — Execute approved proposal
- `GET /dashboard/api/scenarios` — List demo scenarios
- `GET /dashboard/api/ml-stats` — ML model statistics and metrics
- `POST /dashboard/api/seed` — Seed demo data
- `POST /dashboard/api/seed-ml` — Train ML models with demo data
- `POST /dashboard/api/reset` — Clear all demo data, AME tables, and governance gates
- `GET /dashboard/api/workflows` — Dashboard workflow list (DB-backed)
- `POST /dashboard/api/simulate` — Repeat a case N times (load testing / AME training)
- `POST /dashboard/api/auto-step` — Auto-advance workflow using ML predictions
- `POST /dashboard/api/run-demo` — Run demo scenarios with guided walkthrough

### Kroger Demo (`/kroger/*`)

- `GET /kroger` — Redirects to `/dashboard`
- `GET /kroger/scenarios` — List Kroger scenario templates
- `POST /kroger/start` — Start refill workflow
- `POST /kroger/prescriber-approval` — Simulate prescriber approval
- `POST /kroger/submit-preverification` — Push to pre-verification
- `POST /kroger/pharmacist-preverify` — Pharmacist decision

### AME Trust System (`/ame/*`)

- `GET /ame/scope` — Get trust scope status
- `POST /ame/proposal` — Log proposal creation event
- `POST /ame/decision` — Log human decision event
- `POST /ame/execute` — Log execution event
- `POST /ame/outcome` — Log outcome event
- `POST /ame/rollback/{execution_id}` — Rollback a reversible execution
- `POST /ame/resolve-mode` — Resolve execution mode for a scope (debug)
- `GET /ame/scopes` — List all trust scopes
- `GET /ame/events/recent` — Get recent AME events
- `GET /ame/ml/status` — ML model versions and metrics
- `POST /ame/ml/retrain` — Retrain all ML models (admin only)
- `GET /ame/ml/retrain/check` — Check retrain triggers
- `GET /ame/dashboard` — AME trust dashboard (HTML)

## Architecture

### Layered Design

```
FastAPI Routes (main.py + routers)
        ↓
Business Logic (services/ + src/enterprise/ame/)
        ↓
Abstraction Layer (cache.py, storage.py)
        ↓
ORM Models + Pydantic Schemas (models/ + src/enterprise/ame/models.py)
        ↓
config.py (Settings singleton)
        ↓
SQLite / PostgreSQL    Memory / Redis    Local / S3
```

### Router Registration Order

Routers are included in `main.py` in this order (first match wins for overlapping prefixes):

```python
app.include_router(kroger_router)        # /kroger/*
app.include_router(demo_router)          # /demo
app.include_router(dashboard_router)     # /dashboard/*
app.include_router(enterprise_router)    # /enterprise/*
app.include_router(ame_router)           # /ame/*
```

### Key Patterns

- **Unified Config:** `config.py` exposes a frozen `Settings` dataclass derived from environment variables. Import `settings` anywhere via `from config import settings`.
- **Dependency Injection:** FastAPI `Depends()` for DB sessions (`get_db`) and auth (`require_auth`, `require_role`).
- **Demo Auth Bypass:** When `auth_enabled=False` (demo mode), `require_auth` returns a built-in admin `UserContext(user_id="demo", role="admin", key_id="demo0000")` — no API key needed.
- **Cache Abstraction:** `services/cache.py` — `MemoryCache` (demo) or `RedisCache` (prod). Workflow reads are cache-backed with 30s TTL; writes invalidate.
- **Storage Abstraction:** `services/storage.py` — `LocalStorage` (demo) or `S3Storage` (prod). Directory traversal protected.
- **RBAC:** API key auth with timing-safe HMAC comparison via `enterprise/auth.py`.
- **Governance Gates:** Persistent JSON-backed authorization gates (`data/governance_gates.json`) with proposal-approval workflow and full audit trail.
- **Audit-First:** All significant actions logged to `logs/audit/audit_log.jsonl` (JSONL, thread-safe with `threading.Lock()`, auto-rotating at configurable size threshold).
- **Rules Engine:** Condition/action pairs evaluated against workflow context (uses `eval()`/`exec()` — demo only, not production-safe).
- **Multi-DB:** SQLAlchemy abstracts SQLite (NullPool) vs PostgreSQL/MySQL (connection pooling + SSL).
- **Demo Data:** In-memory dicts (`DEMO_ROWS`, `DEMO_PROPOSALS`) in `bladnir_dashboard.py` with negative IDs distinguish demo cases from DB-backed workflows.
- **Auto-Seed:** In demo mode, scenarios are seeded automatically at startup via `app.on_event("startup")` calling `seed_demo_cases()`.
- **Explicit Commits:** The `get_db` dependency does not auto-commit; route handlers must call `db.commit()` explicitly. Rollback is automatic on error.

### Dual Declarative Base (Important Gotcha)

The project uses **two separate SQLAlchemy `Base` classes**:

1. **`models.database.Base`** — Used by `Workflow`, `Task`, `Event`, `Rule` models
2. **`src.enterprise.ame.models.Base`** — Used by `AMETrustScope`, `AMEEvent`, `AMEExecution` models

Both are created at startup in `main.py`:

```python
Base.metadata.create_all(bind=engine)       # Core models
AMEBase.metadata.create_all(bind=engine)    # AME models
```

When adding new ORM models, use the correct Base depending on which subsystem the model belongs to. If creating a model unrelated to AME, inherit from `models.database.Base`.

### Database Models

| Table | Key Fields |
|---|---|
| `workflows` | id, name, description, state (ordered/pending_access/access_granted/dispensed/completed/failed), timestamps |
| `tasks` | id, workflow_id (FK), name, assigned_to, state (pending/in_progress/completed/failed), timestamps |
| `events` | id, workflow_id (FK), event_type, payload (JSON text), created_at |
| `rules` | id, name (unique), description, condition (JSON), action (JSON), enabled, priority, execution_count, timestamps |
| `ame_trust_scopes` | id, tenant_id, site_id, queue, action_type, role_context, stage, trust_score, reliability, alignment, safety_calibration, value_score, override_rate, proposal/execution counters, timestamps |
| `ame_events` | id, ts, scope fields, event_type, proposal_id, predicted metrics, decision fields, outcome fields, features_json, context_json, soft-delete fields |
| `ame_executions` | id, ts, scope fields, proposal_id, before/after state (JSON), diff (JSON), reversible_until, rollback tracking, execution_duration_ms |

State transitions use enums: `WorkflowState`, `TaskState` (defined in `models/schemas.py`), `AMEStage`, `AMEEventType`, `AMEDecision` (defined in `src/enterprise/ame/models.py`).

### File-based Storage

- `data/governance_gates.json` — Governance gate state + change history (JSON, thread-safe writes)
- `logs/audit/audit_log.jsonl` — Audit log (JSONL, auto-rotating at configurable threshold)
- `data/models/{scope_key}/decision_v{n}.pkl` — Decision Predictor model (joblib)
- `data/models/{scope_key}/outcome_v{n}.pkl` — Outcome Predictor model (joblib)
- `data/models/global/anomaly_v{n}.pkl` — Anomaly Detector model (joblib)

**Note:** `data/` and `logs/` directories are `.gitignore`d and created at runtime.

## AME Trust System

The Adaptive Model Evolution (AME) system is a core subsystem that manages **trust-based automation progression**. It tracks how reliably the system performs over time and progressively increases automation levels.

### Trust Stages

```
observe → propose → guarded_auto → conditional_auto → full_auto
```

Each scope (defined by tenant/site/queue/action/role) independently progresses through stages based on trust metrics computed from event history.

### Trust Metrics

- **Reliability:** Approval rate of proposals
- **Alignment:** Success rate of approved executions
- **Safety Calibration:** Accuracy of predicted vs. observed outcomes
- **Value Score:** Time saved relative to manual processing
- **Override Rate:** How often humans override automated decisions

### Trust Score Formula

Trust is a weighted composite with override penalty:

```
trust = (w_reliability * reliability
       + w_alignment * alignment
       + w_safety * safety_calibration
       + w_value * value_score)
       * (1.0 - override_penalty * override_rate)
```

Default weights: reliability=0.35, alignment=0.25, safety=0.25, value=0.15.

### Stage Promotion Thresholds

| Transition | Trust Score Required | Minimum Evidence |
|---|---|---|
| observe → propose | 0.40 | 6 observations |
| propose → guarded_auto | 0.60 | 3 proposals |
| guarded_auto → conditional_auto | 0.75 | 3 executions |
| conditional_auto → full_auto | 0.85 | — |
| Downgrade (any stage) | Below 0.70 | 3 consecutive failures |

Promotion is also blocked if `override_rate > 0.10` or `observed_error_rate > 0.05`.

These thresholds are set low for demo purposes (see `AMEConfig` in `src/enterprise/ame/service.py`).

### ML Models

| Model | Algorithm | Features | Purpose |
|---|---|---|---|
| Decision Predictor | GradientBoostingClassifier | 14 features | Predicts probability of human approval |
| Outcome Predictor | CalibratedClassifierCV(GradientBoosting) | 16 features | Predicts probability of execution success |
| Anomaly Detector | IsolationForest | 9 features | Detects unusual patterns in event streams |

ML models gracefully degrade: when untrained, they return cold-start defaults. Retraining is triggered by event count thresholds, elapsed time, or drift detection.

## Development Conventions

### Code Style

- All modules use `from __future__ import annotations` for forward references
- Extensive docstrings on all public functions and classes
- Type hints on all function signatures using Pydantic and standard typing
- Logging via `logging.getLogger(__name__)` per module
- Thread safety via `threading.Lock()` for shared in-memory state (cache, governance, audit)
- Section separators using `# =====================================` comment blocks

### Adding New Endpoints

1. Create or extend a router in `services/` or `enterprise/`
2. Define Pydantic request/response schemas in `models/schemas.py`
3. Add any new ORM models in `models/schemas.py`, `models/rules.py`, or `src/enterprise/ame/models.py`
4. Register the router in `main.py` via `app.include_router(router)`
5. Use `Depends(get_db)` for database access
6. Use `Depends(require_auth)` or `Depends(require_role("admin"))` for protected endpoints
7. Call `db.commit()` explicitly after mutations — the session does not auto-commit

### Adding New Abstraction Backends

To add a new cache or storage backend:

1. Subclass `CacheBackend` (`services/cache.py`) or `StorageBackend` (`services/storage.py`)
2. Implement all abstract methods
3. Add a new branch in the `_build_cache()` or `_build_storage()` factory
4. Add the corresponding environment variable to `config.py`

### Adding New Database Models

1. Define SQLAlchemy model class inheriting from the correct `Base`:
   - Core models → `from models.database import Base`
   - AME models → `from src.enterprise.ame.models import Base`
2. Define corresponding Pydantic schemas for create/read operations in `models/schemas.py`
3. Tables are auto-created in demo and development modes
4. No migration framework is in place — schema changes require manual handling in production

### Database Session Pattern

```python
from models.database import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

@app.post("/example")
def create_example(db: Session = Depends(get_db)):
    obj = MyModel(name="example")
    db.add(obj)
    db.commit()      # Must commit explicitly
    db.refresh(obj)  # Reload with generated fields
    return obj
```

The `get_db` dependency handles rollback on any exception and always closes the session in `finally`.

## Testing

There are no automated tests. To manually verify changes:

```bash
# Start the server:
uvicorn main:app --reload

# Health check:
curl http://127.0.0.1:8000/health

# List workflows (demo data auto-seeded):
curl http://127.0.0.1:8000/api/workflows

# Dashboard UI:
open http://127.0.0.1:8000/dashboard

# AME trust dashboard:
open http://127.0.0.1:8000/ame/dashboard

# Reset all demo data:
curl -X POST http://127.0.0.1:8000/dashboard/api/reset
```

## Dashboard Features

The dashboard (`services/bladnir_dashboard.py`) is the largest file in the codebase (~2739 lines) and serves as the primary interactive UI. Key capabilities:

- **Mobile-responsive layout** — CSS media queries adapt the UI for mobile and tablet viewports
- **Guided walkthrough / tutorial** — Step-by-step demo tutorial with tooltip positioning and scroll-into-view behavior
- **Live activity narration** — Real-time narrative display of demo actions for investor/stakeholder presentations
- **Industry-specific queues** — Supports Insurance and HR labels alongside the default Kroger pharmacy scenario
- **ML integration** — Inline ML model training (`/dashboard/api/seed-ml`), statistics (`/dashboard/api/ml-stats`), and auto-stepping with ML predictions
- **Demo simulation** — Bulk case simulation (`/dashboard/api/simulate`) for load testing and AME trust training
- **Auto-play demo** — Automated demo run (`/dashboard/api/run-demo`) with scenario sequencing

The dashboard is a single-page application rendered as inline HTML/CSS/JS from Python. There is no separate frontend build step.

## Known Limitations

- **No automated tests** — No test framework or test directory exists
- **No CI/CD** — No GitHub Actions, Jenkinsfile, or similar pipeline
- **No Docker** — No containerization setup
- **Rules engine uses `eval()`/`exec()`** — Unsafe for untrusted input; acceptable for demo purposes only
- **File-based governance persistence** — `data/governance_gates.json` is not database-backed
- **No migration system** — Schema changes are not versioned (no Alembic)
- **Redis/S3 are stubs** — Production backends require `redis` or `boto3` packages not in `requirements.txt`
- **Dashboard UI is embedded HTML/JS** — `bladnir_dashboard.py` returns inline HTML; no separate frontend build
- **Dual Base classes** — Core and AME models use separate declarative bases, requiring both to be created at startup
- **No async database operations** — All DB access is synchronous despite FastAPI's async capability

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'config'` | Running from wrong directory | Run `uvicorn` from the `BladnirTech/` root |
| `AuthenticationError: BLADNIR_ADMIN_KEY not set` | Missing env var in dev/prod mode | Set `BLADNIR_ADMIN_KEY` (32+ chars) or use `ENVIRONMENT=demo` |
| `sqlite3.OperationalError: database is locked` | Concurrent SQLite writes | Expected under load; use PostgreSQL for concurrency |
| Tables not created | Running in production mode | Production mode doesn't auto-create tables; run migrations manually |
| Demo data not appearing | `AUTO_SEED=false` or not in demo mode | Set `ENVIRONMENT=demo` and `AUTO_SEED=true` |
| AME models not created | `AMEBase.metadata.create_all` failed | Check `main.py` startup logs for errors |
| `data/` or `logs/` missing | First run, directories not yet created | Created automatically at runtime; both are `.gitignore`d |

## Recent Changes

Notable changes reflected in the current codebase (most recent first):

- **Industry-specific queues** — Added Insurance/HR queue labels and tasks alongside Kroger pharmacy
- **Mobile-responsive dashboard** — Added CSS media queries for mobile and tablet viewport support
- **Database locking fix** — Removed background threads and reduced SQLite timeout to prevent `database is locked` errors during demo resets
- **Tutorial scroll fix** — Scroll target into view before positioning tooltip during guided walkthrough
- **Investor-ready demo** — De-jargoned UI, cross-industry scenario labels, business narrative framing
- **Live activity narration** — Real-time narration of demo actions during guided walkthrough and auto-play
- **PactGate rebranding** — Renamed from "Control Tower" to "PactGate™" across all UI and documentation

## Security Notes

- Never commit API keys or secrets to the repository
- `BLADNIR_ADMIN_KEY` must be 32+ characters in production
- Authentication uses HMAC-based timing-safe comparison (`hmac.compare_digest`)
- Rate limiting: 10 auth attempts per 5-minute sliding window (in-memory)
- Demo mode auth bypass is **not** active when `ENVIRONMENT=production`
- CORS origins should be restricted to known frontends in production
- The rules engine (`eval`/`exec`) must not be exposed to untrusted user input
- Storage layer prevents directory traversal attacks via path validation
- API keys are SHA256-hashed before storage; raw keys are never logged
- Database credentials are stripped from log output (`DATABASE_URL.split('@')[-1]`)
- Audit log directory created with restrictive permissions (`0o700`)
