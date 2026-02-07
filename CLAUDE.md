# CLAUDE.md — BladnirTech Control Tower

## Project Overview

BladnirTech is a **workflow orchestration middleware platform** built with FastAPI. It provides a "Control Tower" for managing automated workflows with human governance gates, a rules engine, an ML-driven adaptive trust system (AME), and audit-first architecture. The primary demo scenario targets Kroger retail pharmacy prescription refill workflows.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI (ASGI)
- **Server:** Uvicorn 0.29
- **ORM:** SQLAlchemy 2.0+
- **Validation:** Pydantic 2.6+
- **Database:** SQLite (demo/dev), PostgreSQL or MySQL (production)
- **ML:** scikit-learn 1.4+ (GradientBoosting, IsolationForest, CalibratedClassifier)
- **Model Persistence:** joblib 1.3+
- **Numerical:** NumPy 1.26+

## Unified Codebase, Different Environments

The same codebase runs in **demo**, **development**, and **production** modes controlled by the `ENVIRONMENT` variable. Each mode swaps backend implementations via abstraction layers:

| Component | Demo (default) | Production |
|---|---|---|
| **Database** | SQLite | PostgreSQL / MySQL |
| **Cache** | In-memory dict (TTL) | Redis |
| **Storage** | Local filesystem | S3 / Cloud |
| **Auth** | Bypassed (auto-admin) | API key + RBAC |
| **Data seeding** | Auto-seeds on startup | Manual |
| **Table creation** | Automatic | Manual migration |

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
├── config.py                           # Centralized settings (AppMode, env-var parsing)
├── requirements.txt                    # Python dependencies
├── README.md                           # Project readme
├── CLAUDE.md                           # This file
├── .gitignore                          # Ignores __pycache__, *.db, logs/, data/, .env, IDE dirs
├── docs/
│   └── AME_ML_SPEC.md                 # ML design spec for AME trust models
├── enterprise/                         # Enterprise features
│   ├── __init__.py
│   ├── auth.py                         # API key auth, RBAC, rate limiting, demo bypass
│   ├── governance.py                   # Governance gates (proposal-approval flows, JSON persistence)
│   ├── audit.py                        # Audit logging to JSONL files (thread-safe, auto-rotating)
│   └── execute.py                      # Governed execution router (/enterprise/*)
├── models/                             # Data layer
│   ├── __init__.py
│   ├── database.py                     # SQLAlchemy engine, session, multi-DB config
│   ├── schemas.py                      # Pydantic request/response models + state enums
│   └── rules.py                        # Rule ORM model (condition/action pairs)
├── services/                           # Business logic
│   ├── __init__.py
│   ├── workflow.py                     # Workflow/Task/Event ORM models + CRUD (cache-backed reads)
│   ├── rules.py                        # Rules engine (evaluation, CRUD)
│   ├── cache.py                        # Cache abstraction (MemoryCache / RedisCache)
│   ├── storage.py                      # Storage abstraction (LocalStorage / S3Storage)
│   ├── integration.py                  # External system integration stubs (EHR, pharmacy, payer)
│   ├── kroger_retail_pack.py           # Kroger pharmacy demo scenario (/kroger/*)
│   ├── bladnir_dashboard.py            # Dashboard UI + API (/dashboard/*) — largest service file
│   └── demo_hub.py                     # Demo hub landing page (/demo)
└── src/
    └── enterprise/
        └── ame/                        # Adaptive Model Evolution — ML-driven trust system
            ├── __init__.py             # Exports AME router
            ├── models.py               # AMETrustScope, AMEEvent, AMEExecution ORM models
            ├── service.py              # Trust computation, stage management, execution tracking
            ├── router.py               # REST API endpoints (/ame/*)
            └── ml/                     # Machine learning layer
                ├── __init__.py         # Model factory functions
                ├── features.py         # Feature engineering for all 3 ML models
                ├── decision.py         # Approval probability predictor (GradientBoosting)
                ├── outcome.py          # Success probability predictor (CalibratedClassifier)
                ├── anomaly.py          # Anomaly detector (IsolationForest)
                ├── store.py            # Model persistence (joblib, versioned)
                └── trainer.py          # Training orchestration + retrain triggers
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `demo` | `demo`, `development`, or `production` |
| `DATABASE_URL` | `sqlite:///./BladnirTech.db` | Database connection string |
| `BLADNIR_ADMIN_KEY` | (built-in for demo) | Admin API key (min 32 chars, required in prod) |
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

- `GET /dashboard` — Interactive dashboard UI (HTML + embedded JS)
- `GET /dashboard/api/automation` — Get governance gate statuses
- `POST /dashboard/api/automation` — Authorize/revoke governance gates
- `GET /dashboard/api/automation/pending` — List pending proposals
- `GET /dashboard/api/cases/{case_id}` — Get case detail with proposals and trust info
- `POST /dashboard/api/cases/{case_id}/propose` — Propose automated action
- `POST /dashboard/api/automation/{proposal_id}/decide` — Approve/reject proposal
- `POST /dashboard/api/automation/{proposal_id}/execute` — Execute approved proposal
- `GET /dashboard/api/scenarios` — List demo scenarios
- `POST /dashboard/api/seed` — Seed demo data
- `POST /dashboard/api/reset` — Clear all demo data, AME tables, and governance gates
- `GET /dashboard/api/workflows` — Dashboard workflow list (DB-backed)
- `POST /dashboard/api/simulate` — Repeat a case N times (load testing / AME training)
- `POST /dashboard/api/auto-step` — Auto-advance workflow using ML predictions

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

### Key Patterns

- **Unified Config:** `config.py` exposes a frozen `Settings` dataclass derived from environment variables. Import `settings` anywhere.
- **Dependency Injection:** FastAPI `Depends()` for DB sessions and auth.
- **Demo Auth Bypass:** When `auth_enabled=False` (demo mode), `require_auth` returns a built-in admin `UserContext` — no API key needed.
- **Cache Abstraction:** `services/cache.py` — `MemoryCache` (demo) or `RedisCache` (prod). Workflow reads are cache-backed with 30s TTL; writes invalidate.
- **Storage Abstraction:** `services/storage.py` — `LocalStorage` (demo) or `S3Storage` (prod). Directory traversal protected.
- **RBAC:** API key auth with timing-safe HMAC comparison via `enterprise/auth.py`.
- **Governance Gates:** Persistent JSON-backed authorization gates (`data/governance_gates.json`) with proposal-approval workflow and full audit trail.
- **Audit-First:** All significant actions logged to `logs/audit/audit_log.jsonl` (JSONL, thread-safe with `threading.Lock()`, auto-rotating at configurable size threshold).
- **Rules Engine:** Condition/action pairs evaluated against workflow context (uses `eval()`/`exec()` — demo only, not production-safe).
- **Multi-DB:** SQLAlchemy abstracts SQLite (NullPool) vs PostgreSQL/MySQL (connection pooling + SSL).
- **Demo Data:** In-memory dicts (`DEMO_ROWS`, `DEMO_PROPOSALS`) with negative IDs distinguish demo cases from DB-backed workflows.
- **Auto-Seed:** In demo mode, scenarios are seeded automatically at startup via `app.on_event("startup")`.

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

### Adding New Endpoints

1. Create or extend a router in `services/` or `enterprise/`
2. Define Pydantic request/response schemas in `models/schemas.py`
3. Add any new ORM models in `models/schemas.py`, `models/rules.py`, or `src/enterprise/ame/models.py`
4. Register the router in `main.py` via `app.include_router(router)`
5. Use `Depends(get_db)` for database access
6. Use `Depends(require_auth)` or `Depends(require_role("admin"))` for protected endpoints

### Adding New Abstraction Backends

To add a new cache or storage backend:

1. Subclass `CacheBackend` (`services/cache.py`) or `StorageBackend` (`services/storage.py`)
2. Implement all abstract methods
3. Add a new branch in the `_build_cache()` or `_build_storage()` factory
4. Add the corresponding environment variable to `config.py`

### Adding New Database Models

1. Define SQLAlchemy model class inheriting from `Base` (in `models/` or `src/enterprise/ame/models.py`)
2. Define corresponding Pydantic schemas for create/read operations
3. Tables are auto-created in demo and development modes
4. No migration framework is in place — schema changes require manual handling in production

## Known Limitations

- **No automated tests** — No test framework or test directory exists
- **No CI/CD** — No GitHub Actions, Jenkinsfile, or similar pipeline
- **No Docker** — No containerization setup
- **Rules engine uses `eval()`/`exec()`** — Unsafe for untrusted input; acceptable for demo purposes only
- **File-based governance persistence** — `data/governance_gates.json` is not database-backed
- **No migration system** — Schema changes are not versioned
- **Redis/S3 are stubs** — Production backends require `redis` or `boto3` packages not in `requirements.txt`
- **Dashboard UI is embedded HTML/JS** — `bladnir_dashboard.py` returns inline HTML; no separate frontend build

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
