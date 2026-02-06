"""
Centralized application settings with environment-aware defaults.

Determines whether the app runs in demo or production mode and exposes
typed configuration consumed by every other module.  All values fall
back to safe demo defaults so the app starts with zero environment
variables set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class AppMode(str, Enum):
    DEMO = "demo"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def _detect_mode() -> AppMode:
    """Derive the application mode from ENVIRONMENT env-var."""
    raw = os.getenv("ENVIRONMENT", "demo").lower().strip()
    if raw in ("production", "prod"):
        return AppMode.PRODUCTION
    if raw == "development":
        return AppMode.DEVELOPMENT
    return AppMode.DEMO


@dataclass(frozen=True)
class Settings:
    """Immutable, environment-derived settings."""

    # --- mode ---
    mode: AppMode = field(default_factory=_detect_mode)

    # --- database ---
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./BladnirTech.db")
    )

    # --- auth ---
    auth_enabled: bool = field(default_factory=lambda: os.getenv("ENVIRONMENT", "demo").lower() in ("production", "prod"))
    demo_admin_key: str = "demo-admin-key-not-for-production-use!!"

    # --- cache ---
    cache_backend: str = field(
        default_factory=lambda: os.getenv("CACHE_BACKEND", "memory")
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    cache_default_ttl: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL", "300"))
    )

    # --- storage ---
    storage_backend: str = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "local")
    )
    storage_local_dir: str = field(
        default_factory=lambda: os.getenv("STORAGE_LOCAL_DIR", "./data/files")
    )
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "us-east-1"))

    # --- CORS ---
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    )

    # --- audit ---
    audit_log_dir: str = field(
        default_factory=lambda: os.getenv("AUDIT_LOG_DIR", "./logs/audit")
    )
    audit_max_size_mb: int = field(
        default_factory=lambda: int(os.getenv("AUDIT_MAX_SIZE_MB", "100"))
    )

    # --- demo ---
    auto_seed: bool = field(
        default_factory=lambda: os.getenv("AUTO_SEED", "true").lower() in ("1", "true", "yes")
    )
    auto_create_tables: bool = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "demo").lower() != "production"
    )

    # --- convenience helpers ---

    @property
    def is_demo(self) -> bool:
        return self.mode == AppMode.DEMO

    @property
    def is_production(self) -> bool:
        return self.mode == AppMode.PRODUCTION


# Module-level singleton — import ``settings`` from anywhere.
settings = Settings()

logger.info(f"Settings loaded: mode={settings.mode.value}, auth_enabled={settings.auth_enabled}")
