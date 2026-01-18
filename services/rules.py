"""
Rules engine for the PharmAI middleware.

Rules encapsulate payer, plan or clinical logic that determine when certain
workflow transitions or tasks should occur.  Each rule has a `condition`
expression that evaluates to a boolean, and an `action` expression that
executes when the condition is True.  Both expressions are stored as strings
in the database.  When evaluating rules, these expressions are executed
dynamically in a restricted environment.

⚠️ **Security Notice**: Executing code from the database is dangerous.  In a
real implementation you should use a safe expression language or DSL instead
of Python `eval`/`exec`.  This demonstration uses Python for brevity.
"""

from __future__ import annotations

import logging
from typing import List

from sqlalchemy.orm import Session

from models import database
from models.schemas import RuleCreate, RuleRead, WorkflowState

# Create a logger for the rules engine
logger = logging.getLogger(__name__)

# Define ORM models (to be created below) via Base from database
from sqlalchemy import Column, Integer, String, Text


class Rule(database.Base):
    __tablename__ = "rules"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    condition = Column(Text, nullable=False)
    action = Column(Text, nullable=False)


def create_rule(db: Session, rule_in: RuleCreate) -> Rule:
    """Persist a new rule to the database."""
    rule = Rule(
        name=rule_in.name,
        description=rule_in.description,
        condition=rule_in.condition,
        action=rule_in.action,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


def list_rules(db: Session) -> List[Rule]:
    """Return all rules in the system."""
    return db.query(Rule).all()


def evaluate_rules_for_workflow(db: Session, workflow: "Workflow") -> None:
    """Evaluate all rules against a given workflow.

    If a rule's condition evaluates to True, its action is executed.  The
    workflow instance passed in is a SQLAlchemy model from the session; any
    mutations will be persisted when the session commits.
    """
    rules = list_rules(db)
    for rule in rules:
        env = {
            "workflow": workflow,
            "WorkflowState": WorkflowState,
        }
        try:
            if eval(rule.condition, {}, env):
                logger.info("Rule '%s' triggered for workflow %s", rule.name, workflow.id)
                exec(rule.action, {}, env)
        except Exception as e:
            logger.exception("Error evaluating rule '%s': %s", rule.name, e)
