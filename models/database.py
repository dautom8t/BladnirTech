"""
Database configuration and session management.

Supports SQLite (development only) and PostgreSQL (production).
Includes connection pooling, health checks, and proper error handling.
"""

from __future__ import annotations

import logging
import os
from typing import Generator

from sqlalchemy import create_engine, event, pool, exc
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# =====================================
# Configuration
# =====================================

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./BladnirTech.db")
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

# Connection pool settings (ignored for SQLite)
POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour

# =====================================
# Validation
# =====================================

def _validate_database_url(url: str) -> None:
    """
    Validate database URL for security and compatibility.
    
    Raises:
        ValueError: If URL is invalid or insecure
    """
    if not url:
        raise ValueError("DATABASE_URL cannot be empty")
    
    # Check for allowed database types
    allowed_schemes = ["sqlite", "postgresql", "postgresql+psycopg2", "mysql", "mysql+pymysql"]
    if not any(url.startswith(scheme + "://") or url.startswith(scheme + ":///") 
               for scheme in allowed_schemes):
        raise ValueError(
            f"Unsupported database URL scheme. Allowed: {', '.join(allowed_schemes)}"
        )
    
    # Warn about SQLite in production
    if url.startswith("sqlite") and ENVIRONMENT == "production":
        logger.critical(
            "⚠️  CRITICAL: SQLite detected in production environment! "
            "SQLite is NOT suitable for production. Use PostgreSQL or MySQL."
        )
        # Optionally: raise ValueError("SQLite not allowed in production")


def _get_engine_config(url: str) -> dict:
    """
    Get engine configuration based on database type and environment.
    
    Args:
        url: Database connection URL
        
    Returns:
        Dict of engine configuration parameters
    """
    config = {
        "echo": ENVIRONMENT == "development",  # SQL logging in dev only
        "future": True,  # Use SQLAlchemy 2.0 API
    }
    
    if url.startswith("sqlite"):
        # SQLite-specific configuration
        config.update({
            "connect_args": {
                "check_same_thread": False,  # Required for FastAPI
                "timeout": 20.0,  # Wait up to 20s for locks
            },
            # Use NullPool for SQLite to avoid connection issues
            "poolclass": NullPool,
        })
        logger.warning(
            "Using SQLite database. This is suitable for development only. "
            "For production, use PostgreSQL or MySQL."
        )
    
    elif url.startswith("postgresql"):
        # PostgreSQL-specific configuration
        config.update({
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_timeout": POOL_TIMEOUT,
            "pool_recycle": POOL_RECYCLE,
            "pool_pre_ping": True,  # Test connections before using
        })
        
        # SSL in production
        if ENVIRONMENT == "production":
            config["connect_args"] = {
                "sslmode": "require",  # Require SSL
                "connect_timeout": 10,
            }
        
        logger.info(
            f"PostgreSQL connection pool configured: "
            f"size={POOL_SIZE}, max_overflow={MAX_OVERFLOW}"
        )
    
    elif url.startswith("mysql"):
        # MySQL-specific configuration
        config.update({
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_timeout": POOL_TIMEOUT,
            "pool_recycle": POOL_RECYCLE,
            "pool_pre_ping": True,
        })
        
        if ENVIRONMENT == "production":
            config["connect_args"] = {
                "ssl": {"ssl_mode": "REQUIRED"},
                "connect_timeout": 10,
            }
        
        logger.info(
            f"MySQL connection pool configured: "
            f"size={POOL_SIZE}, max_overflow={MAX_OVERFLOW}"
        )
    
    return config


# =====================================
# Engine Setup
# =====================================

# Validate URL before creating engine
_validate_database_url(DATABASE_URL)

# Create engine with appropriate configuration
try:
    engine = create_engine(DATABASE_URL, **_get_engine_config(DATABASE_URL))
    logger.info(f"Database engine created successfully: {DATABASE_URL.split('@')[-1]}")  # Hide credentials
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise


# =====================================
# Connection Event Handlers
# =====================================

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log when a new connection is established."""
    logger.debug("New database connection established")


@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Log when a connection is closed."""
    logger.debug("Database connection closed")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Called when a connection is retrieved from the pool."""
    # Optional: Add custom connection setup here
    pass


# =====================================
# Session Configuration
# =====================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Allow accessing objects after commit
)

# Declarative Base
Base = declarative_base()


# =====================================
# Dependency Injection
# =====================================

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Provides a database session with automatic:
    - Connection management
    - Rollback on error
    - Cleanup in finally block
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        SQLAlchemy Session
    """
    db: Session = SessionLocal()
    try:
        yield db
        # Commit is now explicit in route handlers
    except exc.SQLAlchemyError as e:
        logger.error(f"Database error: {e}", exc_info=True)
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


# =====================================
# Health Check
# =====================================

def check_database_health() -> bool:
    """
    Check if database connection is healthy.
    
    Returns:
        True if database is accessible, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_connection_pool_status() -> dict:
    """
    Get current connection pool statistics.
    
    Returns:
        Dict with pool metrics (empty for SQLite/NullPool)
    """
    if isinstance(engine.pool, NullPool):
        return {"pool_type": "NullPool (SQLite)", "status": "N/A"}
    
    try:
        return {
            "pool_type": engine.pool.__class__.__name__,
            "size": engine.pool.size(),
            "checked_in": engine.pool.checkedin(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
            "max_overflow": engine.pool._max_overflow,
        }
    except Exception as e:
        logger.error(f"Failed to get pool status: {e}")
        return {"error": str(e)}


# =====================================
# Startup Logging
# =====================================

logger.info(f"Database module initialized")
logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Database type: {DATABASE_URL.split('://')[0]}")

if ENVIRONMENT == "production" and DATABASE_URL.startswith("sqlite"):
    logger.critical(
        "⚠️  PRODUCTION ENVIRONMENT WITH SQLITE DATABASE ⚠️\n"
        "This is a critical misconfiguration. SQLite should only be used in development.\n"
        "Please configure DATABASE_URL to use PostgreSQL or MySQL in production."
    )
