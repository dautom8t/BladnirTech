# CLAUDE.md — BladnirTech Control Tower

## Project Overview

BladnirTech is a **workflow orchestration middleware platform** built with FastAPI. It provides a "Control Tower" for managing automated workflows with human governance gates, a rules engine, audit-first architecture, and an **Adaptive Model Evolution (AME)** trust system with machine learning. The primary demo scenario targets Kroger retail pharmacy prescription refill workflows.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI (ASGI)
- **Server:** Uvicorn 0.29
- **ORM:** SQLAlchemy 2.0+
- **Validation:** Pydantic 2.6+
- **Database:** SQLite (demo/dev), PostgreSQL or MySQL (production)
- **ML:** scikit-learn 1.4+, joblib 1.3+, numpy 1.26+ (graceful degradation if unavailable)

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
| **ML Models** | Cold-start defaults | Trained on accumulated data |

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
├── main.py                         # FastAPI app entry point, core API routes, startup hooks
├── config.py                       # Centralized settings (AppMode, env-var parsing)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignores __pycache__, *.db, logs/, data/, .env, IDE configs
├── README.md                       # Project readme
├── CLAUDE.md                       # This file
├── docs/
│   └── AME_ML_SPEC.md             # ML layer specification for AME trust system
├── enterprise/                     # Enterprise features
│   ├── __init__.py
│   ├── auth.py                     # API key auth, RBAC, rate limiting, demo bypass
│   ├── governance.py               # Governance gates (proposal-approval flows, JSON-backed)
│   ├── audit.py                    # Audit logging to JSONL files (thread-safe, rotating)
│   └── execute.py                  # Governed execution router (/enterprise/*)
├── models/                         # Data layer
│   ├── __init__.py
│   ├── database.py                 # SQLAlchemy engine, session, multi-DB config
│   ├── schemas.py                  # Pydantic request/response models + ORM models
│   └── rules.py                    # Rule ORM model (condition/action pairs)
├── services/                       # Business logic
│   ├── __init__.py
│   ├── workflow.py                 # Workflow CRUD (cache-backed reads) + ORM models
│   ├── rules.py                    # Rules engine (evaluation, CRUD)
│   ├── cache.py                    # Cache abstraction (memory / Redis)
│   ├── storage.py                  # Storage abstraction (local / S3)
│   ├── integration.py              # External system integration helpers (EHR, pharmacy, payer)
│   ├── kroger_retail_pack.py       # Kroger pharmacy demo scenario (/kroger/*), AME-integrated
│   ├── bladnir_dashboard.py        # Dashboard UI + API (/dashboard/*), AME-integrated
│   └── demo_hub.py                 # Demo hub landing page (/demo)
└── src/                            # Extended enterprise features
    └── enterprise/
        └── ame/                    # Adaptive Model Evolution trust system
            ├── __init__.py         # Exports ame_router
            ├── models.py           # AME ORM models (AMETrustScope, AMEEvent, AMEExecution)
            ├── service.py          # Trust computation, stage management, execution tracking
            ├── router.py           # AME API endpoints (/ame/*)
            └── ml/                 # Machine learning layer
                ├── __init__.py     # Model caches + lazy loaders
                ├── decision.py     # Decision Predictor (will human approve?)
                ├── outcome.py      # Outcome Predictor (will action succeed?)
                ├── anomaly.py      # Anomaly Detector (is event sequence normal?)
                ├── features.py     # Feature engineering for all ML models
                ├── trainer.py      # Training orchestration for all models
                └── store.py        # Model persistence to disk (joblib + metadata)
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

### AME Trust System (`/ame/*`)

- `POST /ame/propose` — Log AI proposal event with predictions
- `POST /ame/decide` — Log human decision (approve/reject/defer)
- `POST /ame/execute` — Create reversible execution with rollback window
- `POST /ame/outcome` — Log observed outcome (success/failure)
- `POST /ame/mode` — Resolve execution mode for a given context (observe/propose/auto)
- `GET /ame/scopes` — List all trust scopes with metrics
- `GET /ame/scopes/{scope_key}` — Get detailed scope status
- `GET /ame/events` — Query events with filtering
- `POST /ame/ml/train` — Trigger ML model training
- `GET /ame/ml/models` — Get trained model status and metadata

### Dashboard (`/dashboard/*`)

- `GET /dashboard` — Interactive dashboard UI
- `GET /dashboard/api/workflows` — Dashboard workflow list
- `POST /dashboard/api/seed` — Seed demo data
- `POST /dashboard/api/cases/{id}/propose` — Propose automated action (AME-integrated)
- `POST /dashboard/api/automation/{id}/decide` — Approve/reject proposal (AME-integrated)
- `POST /dashboard/api/automation/{id}/execute` — Execute approved proposal (AME-integrated)

### Kroger Demo (`/kroger/*`)

- `GET /kroger` — Interactive Kroger demo UI
- `POST /kroger/start` — Start refill workflow
- `POST /kroger/prescriber-approval` — Simulate prescriber approval
- `POST /kroger/submit-preverification` — Push to pre-verification
- `POST /kroger/pharmacist-preverify` — Pharmacist decision

## Architecture

### Layered Design

```
FastAPI Routes (main.py + routers)
        ↓
Business Logic (services/ + src/enterprise/ame/)
        ↓
AME Trust Layer (trust scoring, ML predictions, stage management)
        ↓
Abstraction Layer (cache.py, storage.py)
        ↓
ORM Models + Pydantic Schemas (models/ + src/enterprise/ame/models.py)
        ↓
config.py (Settings singleton)
        ↓
SQLite / PostgreSQL    Memory / Redis    Local / S3    ML Models (joblib)
```

### Key Patterns

- **Unified Config:** `config.py` exposes a frozen `Settings` dataclass derived from environment variables. Import `settings` anywhere.
- **Dependency Injection:** FastAPI `Depends()` for DB sessions and auth.
- **Demo Auth Bypass:** When `auth_enabled=False` (demo mode), `require_auth` returns a built-in admin `UserContext` — no API key needed.
- **Cache Abstraction:** `services/cache.py` — `MemoryCache` (demo) or `RedisCache` (prod). Workflow reads are cache-backed with 30s TTL; writes invalidate.
- **Storage Abstraction:** `services/storage.py` — `LocalStorage` (demo) or `S3Storage` (prod). Directory traversal protected.
- **RBAC:** API key auth with timing-safe comparison (HMAC) via `enterprise/auth.py`. Includes rate limiting (5 attempts per 5 min).
- **Governance Gates:** Persistent JSON-backed authorization gates (`data/governance_gates.json`) with proposal-approval workflow, full change history, and emergency reset.
- **Audit-First:** All significant actions logged to `logs/audit/audit_log.jsonl` (JSONL, thread-safe, auto-rotating). Async logging available via `ThreadPoolExecutor`.
- **Rules Engine:** Condition/action pairs evaluated against workflow context (uses `eval()`/`exec()` — demo only, not production-safe).
- **Multi-DB:** SQLAlchemy abstracts SQLite (NullPool) vs PostgreSQL/MySQL (connection pooling + SSL).
- **Demo Workflows:** In-memory workflows use negative IDs; DB workflows use positive IDs.
- **Auto-Seed:** In demo mode, all scenarios are seeded automatically at startup via `app.on_event("startup")`.
- **Non-Blocking AME:** Dashboard and Kroger routes wrap AME calls in `_ame_safe()` — AME failures log warnings but never break workflow execution.
- **Dual ORM Base:** `src/enterprise/ame/models.py` uses the shared `Base` from `models/database.py`. Both core and AME tables are created at startup.
- **Graceful ML Degradation:** ML features check `HAS_SKLEARN` flag; cold-start defaults are returned when models are not trained or scikit-learn is unavailable.

### Database Models

#### Core Tables (models/)

| Table | Key Fields |
|---|---|
| `workflows` | id, name, description, state (ordered/pending_access/access_granted/dispensed/completed/failed), timestamps |
| `tasks` | id, workflow_id (FK), name, assigned_to, state (pending/in_progress/completed/failed), timestamps |
| `events` | id, workflow_id (FK), event_type, payload (JSON), created_at |
| `rules` | id, name, description, condition (JSON), action (JSON), enabled, priority, execution_count, last_executed_at |

State transitions use enums: `WorkflowState`, `TaskState` (defined in `models/schemas.py`).

#### AME Trust Tables (src/enterprise/ame/models.py)

| Table | Key Fields |
|---|---|
| `ame_trust_scopes` | id, tenant_id, site_id, queue, action_type, role_context, stage (observe/propose/guarded_auto/conditional_auto/full_auto), trust_score, reliability, alignment, safety_calibration, value_score, override_rate, proposal/execution/rollback counts, stage change tracking |
| `ame_events` | id, ts, tenant_id, site_id, queue, action_type, role_context, event_type, proposal_id, predicted_confidence/safety/time_saved, decision (approve/reject/defer), outcome_success, observed_error, features_json, context_json, metadata_json |
| `ame_executions` | id, ts, tenant_id, site_id, queue, action_type, role_context, proposal_id, before_state_json, after_state_json, diff_json, reversible_until, rolled_back, rollback tracking |

`ame_trust_scopes` has a unique constraint on `(tenant_id, site_id, queue, action_type, role_context)`.

### AME Trust System

The Adaptive Model Evolution (AME) system tracks trust per scope (tenant + site + queue + action + role) and progressively grants automation autonomy:

```
OBSERVE → PROPOSE → GUARDED_AUTO → CONDITIONAL_AUTO → FULL_AUTO
```

**Trust scoring** uses weighted metrics (configurable in `AMEConfig`):
- Reliability (35%): successful executions / total
- Alignment (25%): approved proposals / total decisions
- Safety Calibration (25%): inverse of prediction error
- Value Score (15%): time saved by automation

**Stage thresholds** (default): 0.40 → 0.60 → 0.75 → 0.85

**Reversible executions:** At `GUARDED_AUTO`+, executions capture before/after state snapshots with time-limited rollback windows.

### ML Layer (src/enterprise/ame/ml/)

Three trained models sit on top of the AME event data:

| Model | Question | Algorithm | Cold-Start Default |
|---|---|---|---|
| **Decision Predictor** | Will a human approve this proposal? | GradientBoostingClassifier | 0.5 probability |
| **Outcome Predictor** | Will this action succeed? | GradientBoostingClassifier + CalibratedClassifierCV | 0.86 success prob |
| **Anomaly Detector** | Is this event sequence normal? | IsolationForest | No anomaly detected |

**Feature engineering** (`features.py`) extracts 13+ features per model from `AMEEvent` and `AMETrustScope` data, including queue, action type, scope trust, approval rate, rejection streak, hour of day, and insurance result.

**Model persistence** (`store.py`): Models are saved to `data/models/{scope_key}/` via joblib with JSON metadata (version, metrics, training event count).

**Training** is triggered via `POST /ame/ml/train` or the `AMETrainer` class. Minimum training data requirements: 50 decided proposals for decision, 30 outcomes for outcome, 100 events for anomaly.

See `docs/AME_ML_SPEC.md` for the full ML specification.

## Development Conventions

### Code Style

- All modules use `from __future__ import annotations` for forward references
- Extensive docstrings on all public functions and classes
- Type hints on all function signatures using Pydantic and standard typing
- Logging via `logging.getLogger(__name__)` per module

### Adding New Endpoints

1. Create or extend a router in `services/`, `enterprise/`, or `src/enterprise/`
2. Define Pydantic request/response schemas (in `models/schemas.py` for core, or inline in routers for domain-specific)
3. Add any new ORM models in `models/` (core) or `src/enterprise/ame/models.py` (AME)
4. Register the router in `main.py` via `app.include_router(router)`
5. Use `Depends(get_db)` for database access
6. Use `Depends(require_auth)` or `Depends(require_role("admin"))` for protected endpoints
7. For AME-integrated features, wrap AME calls with non-blocking error handling (see `_ame_safe()` pattern in dashboard/kroger)

### Adding New Abstraction Backends

To add a new cache or storage backend:

1. Subclass `CacheBackend` (`services/cache.py`) or `StorageBackend` (`services/storage.py`)
2. Implement all abstract methods
3. Add a new branch in the `_build_cache()` or `_build_storage()` factory
4. Add the corresponding environment variable to `config.py`

### Adding New Database Models

1. Define SQLAlchemy model class inheriting from `Base` (from `models/database.py`)
2. Define corresponding Pydantic schemas for create/read operations
3. Tables are auto-created in demo and development modes
4. No migration framework is in place — schema changes require manual handling in production

### Adding New ML Models

1. Create a new predictor class in `src/enterprise/ame/ml/` following the pattern of `decision.py` or `outcome.py`
2. Add feature extraction functions in `features.py`
3. Add training data extraction function in `features.py` (query from `AMEEvent`)
4. Register the new model in `trainer.py` for orchestrated training
5. Add a lazy loader in `ml/__init__.py` for caching
6. Handle cold-start gracefully (return sensible defaults when no model is trained)

## Known Limitations

- **No automated tests** — No test framework or test directory exists
- **No CI/CD** — No GitHub Actions, Jenkinsfile, or similar pipeline
- **No Docker** — No containerization setup
- **Rules engine uses `eval()`/`exec()`** — Unsafe for untrusted input; acceptable for demo purposes only
- **File-based governance persistence** — `data/governance_gates.json` is not database-backed
- **No migration system** — Schema changes are not versioned
- **Redis/S3 are stubs** — Production backends require `redis` or `boto3` packages not in `requirements.txt`
- **ML retrain scheduler not implemented** — Training must be triggered manually via API; `docs/AME_ML_SPEC.md` describes a planned scheduler
- **ML models are scope-specific** — Each trust scope trains its own decision/outcome models; no cross-scope transfer learning

## Security Notes

- Never commit API keys or secrets to the repository
- `BLADNIR_ADMIN_KEY` must be 32+ characters in production
- Authentication uses HMAC-based timing-safe comparison
- Demo mode auth bypass is **not** active when `ENVIRONMENT=production`
- CORS origins should be restricted to known frontends in production
- The rules engine (`eval`/`exec`) must not be exposed to untrusted user input
- Storage layer prevents directory traversal attacks
- ML model files (`data/models/`) are excluded from git via `.gitignore`; do not commit trained models containing potentially sensitive patterns
