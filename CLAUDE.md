# CLAUDE.md — BladnirTech Control Tower

## Project Overview

BladnirTech is a **workflow orchestration middleware platform** built with FastAPI. It provides a "Control Tower" for managing automated workflows with human governance gates, a rules engine, and audit-first architecture. The primary demo scenario targets Kroger retail pharmacy prescription refill workflows.

## Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI (ASGI)
- **Server:** Uvicorn 0.29
- **ORM:** SQLAlchemy 2.0+
- **Validation:** Pydantic 2.6+
- **Database:** SQLite (dev), PostgreSQL or MySQL (production)

## Repository Structure

```
BladnirTech/
├── main.py                         # FastAPI app entry point, core API routes
├── requirements.txt                # Python dependencies
├── README.md                       # Project readme
├── enterprise/                     # Enterprise features
│   ├── auth.py                     # API key auth, RBAC, rate limiting
│   ├── governance.py               # Governance gates (proposal-approval flows)
│   ├── audit.py                    # Audit logging to JSONL files
│   └── execute.py                  # Governed execution router (/enterprise/*)
├── models/                         # Data layer
│   ├── database.py                 # SQLAlchemy engine, session, multi-DB config
│   ├── schemas.py                  # Pydantic request/response models + ORM models
│   └── rules.py                    # Rule ORM model (condition/action pairs)
└── services/                       # Business logic
    ├── workflow.py                  # Workflow CRUD operations
    ├── rules.py                    # Rules engine (evaluation, CRUD)
    ├── integration.py              # External system integration helpers
    ├── kroger_retail_pack.py       # Kroger pharmacy demo scenario (/kroger/*)
    ├── bladnir_dashboard.py        # Dashboard UI + API (/dashboard/*)
    └── demo_hub.py                 # Demo hub landing page (/demo)
```

## Running the Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Development

```bash
ENVIRONMENT=development uvicorn main:app --reload
```

Setting `ENVIRONMENT=development` enables auto-creation of database tables via `Base.metadata.create_all()`.

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | (none) | Set to `development` to auto-create DB tables |
| `DATABASE_URL` | `sqlite:///./BladnirTech.db` | Database connection string |
| `BLADNIR_ADMIN_KEY` | (required in prod) | Admin API key (min 32 chars) |
| `BLADNIR_KEY_<NAME>` | (optional) | Service account keys, format `key:role` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `DB_POOL_SIZE` | `5` | Connection pool size (PostgreSQL/MySQL) |
| `DB_MAX_OVERFLOW` | `10` | Max pool overflow connections |
| `DB_POOL_TIMEOUT` | `30` | Pool timeout in seconds |
| `DB_POOL_RECYCLE` | `3600` | Connection recycle time in seconds |
| `AUDIT_LOG_DIR` | `./logs/audit` | Audit log directory |
| `AUDIT_MAX_SIZE_MB` | `100` | Audit log rotation threshold |

## Key API Endpoints

### Core Workflow API (`main.py`)

- `GET /health` — Health check
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
ORM Models + Pydantic Schemas (models/)
        ↓
SQLAlchemy Engine → SQLite / PostgreSQL / MySQL
```

### Key Patterns

- **Dependency Injection:** FastAPI `Depends()` for DB sessions and auth
- **RBAC:** API key auth with timing-safe comparison (HMAC) via `enterprise/auth.py`
- **Governance Gates:** Persistent JSON-backed authorization gates with proposal-approval workflow
- **Audit-First:** All significant actions logged to `logs/audit/audit_log.jsonl` (JSONL, thread-safe, auto-rotating)
- **Rules Engine:** Condition/action pairs evaluated against workflow context (uses `eval()`/`exec()` — demo only, not production-safe)
- **Multi-DB:** SQLAlchemy abstracts SQLite (NullPool) vs PostgreSQL/MySQL (connection pooling + SSL)
- **Demo Workflows:** In-memory workflows use negative IDs; DB workflows use positive IDs

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

### Adding New Database Models

1. Define SQLAlchemy model class inheriting from `Base` (in `models/`)
2. Define corresponding Pydantic schemas for create/read operations
3. Tables are auto-created in development when `ENVIRONMENT=development`
4. No migration framework is in place — schema changes require manual handling in production

## Known Limitations

- **No automated tests** — No test framework or test directory exists
- **No CI/CD** — No GitHub Actions, Jenkinsfile, or similar pipeline
- **No Docker** — No containerization setup
- **No `.gitignore`** — All files are tracked; be cautious about committing secrets or build artifacts
- **Rules engine uses `eval()`/`exec()`** — Unsafe for untrusted input; acceptable for demo purposes only
- **File-based governance persistence** — `data/governance_gates.json` is not database-backed
- **No migration system** — Schema changes are not versioned

## Security Notes

- Never commit API keys or secrets to the repository
- `BLADNIR_ADMIN_KEY` must be 32+ characters in production
- Authentication uses HMAC-based timing-safe comparison
- CORS origins should be restricted to known frontends in production
- The rules engine (`eval`/`exec`) must not be exposed to untrusted user input
