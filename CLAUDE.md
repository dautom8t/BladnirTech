# CLAUDE.md — BladnirTech Control Tower

## Project Overview

BladnirTech is a **workflow orchestration middleware platform** built with FastAPI. It provides a "Control Tower" for managing automated workflows with human governance gates, a rules engine, and audit-first architecture. The primary demo scenario targets Kroger retail pharmacy prescription refill workflows.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI (ASGI)
- **Server:** Uvicorn 0.29
- **ORM:** SQLAlchemy 2.0+
- **Validation:** Pydantic 2.6+
- **Database:** SQLite (demo/dev), PostgreSQL or MySQL (production)

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
├── main.py                         # FastAPI app entry point, core API routes, startup hooks
├── config.py                       # Centralized settings (AppMode, env-var parsing)
├── requirements.txt                # Python dependencies
├── README.md                       # Project readme
├── CLAUDE.md                       # This file
├── enterprise/                     # Enterprise features
│   ├── auth.py                     # API key auth, RBAC, rate limiting, demo bypass
│   ├── governance.py               # Governance gates (proposal-approval flows)
│   ├── audit.py                    # Audit logging to JSONL files
│   └── execute.py                  # Governed execution router (/enterprise/*)
├── models/                         # Data layer
│   ├── database.py                 # SQLAlchemy engine, session, multi-DB config
│   ├── schemas.py                  # Pydantic request/response models + ORM models
│   └── rules.py                    # Rule ORM model (condition/action pairs)
└── services/                       # Business logic
    ├── workflow.py                  # Workflow CRUD (cache-backed reads)
    ├── rules.py                    # Rules engine (evaluation, CRUD)
    ├── cache.py                    # Cache abstraction (memory / Redis)
    ├── storage.py                  # Storage abstraction (local / S3)
    ├── integration.py              # External system integration helpers
    ├── kroger_retail_pack.py       # Kroger pharmacy demo scenario (/kroger/*)
    ├── bladnir_dashboard.py        # Dashboard UI + API (/dashboard/*)
    └── demo_hub.py                 # Demo hub landing page (/demo)
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

### Dashboard (`/dashboard/*`)

- `GET /dashboard` — Interactive dashboard UI
- `GET /dashboard/api/workflows` — Dashboard workflow list
- `POST /dashboard/api/seed` — Seed demo data
- `POST /dashboard/api/cases/{id}/propose` — Propose automated action
- `POST /dashboard/api/automation/{id}/decide` — Approve/reject proposal
- `POST /dashboard/api/automation/{id}/execute` — Execute approved proposal

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
Business Logic (services/)
        ↓
Abstraction Layer (cache.py, storage.py)
        ↓
ORM Models + Pydantic Schemas (models/)
        ↓
config.py (Settings singleton)
        ↓
SQLite / PostgreSQL    Memory / Redis    Local / S3
```

### Key Patterns

- **Unified Config:** `config.py` exposes a frozen `Settings` dataclass derived from environment variables. Import `settings` anywhere.
- **Dependency Injection:** FastAPI `Depends()` for DB sessions and auth
- **Demo Auth Bypass:** When `auth_enabled=False` (demo mode), `require_auth` returns a built-in admin `UserContext` — no API key needed
- **Cache Abstraction:** `services/cache.py` — `MemoryCache` (demo) or `RedisCache` (prod). Workflow reads are cache-backed with 30s TTL; writes invalidate.
- **Storage Abstraction:** `services/storage.py` — `LocalStorage` (demo) or `S3Storage` (prod). Directory traversal protected.
- **RBAC:** API key auth with timing-safe comparison (HMAC) via `enterprise/auth.py`
- **Governance Gates:** Persistent JSON-backed authorization gates with proposal-approval workflow
- **Audit-First:** All significant actions logged to `logs/audit/audit_log.jsonl` (JSONL, thread-safe, auto-rotating)
- **Rules Engine:** Condition/action pairs evaluated against workflow context (uses `eval()`/`exec()` — demo only, not production-safe)
- **Multi-DB:** SQLAlchemy abstracts SQLite (NullPool) vs PostgreSQL/MySQL (connection pooling + SSL)
- **Demo Workflows:** In-memory workflows use negative IDs; DB workflows use positive IDs
- **Auto-Seed:** In demo mode, all scenarios are seeded automatically at startup via `app.on_event("startup")`

### Database Models

| Table | Key Fields |
|---|---|
| `workflows` | id, name, description, state (pending/active/completed/failed), timestamps |
| `tasks` | id, workflow_id (FK), name, assigned_to, state (pending/in_progress/completed/failed) |
| `events` | id, workflow_id (FK), event_type, payload (JSON) |
| `rules` | id, name, condition (JSON), action (JSON), enabled, priority |

State transitions use enums: `WorkflowState`, `TaskState` (defined in `models/schemas.py`).

## Development Conventions

### Code Style

- All modules use `from __future__ import annotations` for forward references
- Extensive docstrings on all public functions and classes
- Type hints on all function signatures using Pydantic and standard typing
- Logging via `logging.getLogger(__name__)` per module

### Adding New Endpoints

1. Create or extend a router in `services/` or `enterprise/`
2. Define Pydantic request/response schemas in `models/schemas.py`
3. Add any new ORM models in `models/schemas.py` or `models/rules.py`
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

1. Define SQLAlchemy model class inheriting from `Base` (in `models/`)
2. Define corresponding Pydantic schemas for create/read operations
3. Tables are auto-created in demo and development modes
4. No migration framework is in place — schema changes require manual handling in production

## Known Limitations

- **No automated tests** — No test framework or test directory exists
- **No CI/CD** — No GitHub Actions, Jenkinsfile, or similar pipeline
- **No Docker** — No containerization setup
- **No `.gitignore`** — All files are tracked; be cautious about committing secrets or build artifacts
- **Rules engine uses `eval()`/`exec()`** — Unsafe for untrusted input; acceptable for demo purposes only
- **File-based governance persistence** — `data/governance_gates.json` is not database-backed
- **No migration system** — Schema changes are not versioned
- **Redis/S3 are stubs** — Production backends require `redis` or `boto3` packages not in `requirements.txt`

## Security Notes

- Never commit API keys or secrets to the repository
- `BLADNIR_ADMIN_KEY` must be 32+ characters in production
- Authentication uses HMAC-based timing-safe comparison
- Demo mode auth bypass is **not** active when `ENVIRONMENT=production`
- CORS origins should be restricted to known frontends in production
- The rules engine (`eval`/`exec`) must not be exposed to untrusted user input
- Storage layer prevents directory traversal attacks
