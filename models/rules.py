"""
SQLAlchemy ORM model for Rules.

Rules define automated workflows based on conditions and actions.
They use a safe, structured format (JSON) instead of arbitrary code execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, JSON, Index
from sqlalchemy.orm import validates

from models.database import Base


class Rule(Base):
    """
    Business rule for workflow automation.
    
    Rules consist of:
    - Condition: JSON structure defining when the rule triggers
    - Action: JSON structure defining what happens when triggered
    
    Example rule:
    {
        "condition": {
            "type": "workflow_state",
            "state": "data_entry",
            "all": [
                {"field": "insurance_verified", "operator": "equals", "value": true},
                {"field": "data_complete", "operator": "equals", "value": true}
            ]
        },
        "action": {
            "type": "advance_queue",
            "to_queue": "pre_verification",
            "notify": ["pharmacy_manager@example.com"]
        }
    }
    """
    
    __tablename__ = "rules"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Metadata
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Rule definition (structured JSON, not arbitrary code)
    condition = Column(JSON, nullable=False, comment="Structured condition definition")
    action = Column(JSON, nullable=False, comment="Structured action definition")
    
    # Status
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    
    # Priority (lower number = higher priority)
    priority = Column(Integer, nullable=False, default=100, index=True)
    
    # Audit fields
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=True)
    updated_by = Column(String(100), nullable=True)
    
    # Execution tracking
    execution_count = Column(Integer, nullable=False, default=0)
    last_executed_at = Column(DateTime, nullable=True)
    last_execution_status = Column(String(50), nullable=True)  # "success", "failure", "skipped"
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_rules_enabled_priority', 'enabled', 'priority'),
    )
    
    @validates('name')
    def validate_name(self, key, value):
        """Validate rule name."""
        if not value or not value.strip():
            raise ValueError("Rule name cannot be empty")
        if len(value) > 200:
            raise ValueError("Rule name must be 200 characters or less")
        return value.strip()
    
    @validates('condition')
    def validate_condition(self, key, value):
        """Validate condition structure."""
        if not isinstance(value, dict):
            raise ValueError("Condition must be a dictionary")
        if 'type' not in value:
            raise ValueError("Condition must have a 'type' field")
        # Add more validation based on your condition types
        return value
    
    @validates('action')
    def validate_action(self, key, value):
        """Validate action structure."""
        if not isinstance(value, dict):
            raise ValueError("Action must be a dictionary")
        if 'type' not in value:
            raise ValueError("Action must have a 'type' field")
        # Add more validation based on your action types
        return value
    
    def __repr__(self):
        return f"<Rule(id={self.id}, name='{self.name}', enabled={self.enabled})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "action": self.action,
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "execution_count": self.execution_count,
            "last_executed_at": self.last_executed_at.isoformat() if self.last_executed_at else None,
            "last_execution_status": self.last_execution_status,
        }
