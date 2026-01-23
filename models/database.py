"""
Database configuration and session management.

This module sets up a simple SQLAlchemy engine pointing at an SQLite database by default.
You can switch the database URL via the `DATABASE_URL` environment variable.

The sessionmaker pattern is used to provide dependency‑injected sessions in FastAPI endpoints.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./pharmAI.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
