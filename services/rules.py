"""
Rules engine.

Stores rules in the database and evaluates them against workflows.
Rules use a safe JSON-based DSL — no eval()/exec().

Condition format (JSON object):
  Simple:   {"field": "workflow.state", "op": "eq", "value": "ordered"}
  Compound: {"all": [...conditions...]} or {"any": [...conditions...]}

Action format (JSON object):
  {"type": "set_workflow_state", "value": "pending_access"}

Supported operators: eq, ne, gt, lt, gte, lte, in, not_in, contains, starts_with
Supported fields:    workflow.state, workflow.name, workflow.description
Supported actions:   set_workflow_state
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from models.schemas import RuleCreate, WorkflowState
from models import database
from models.rules import Rule

logger = logging.getLogger(__name__)


def create_rule(db: Session, rule_in: RuleCreate) -> Rule:
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


def list_rules(db: Session):
    return db.query(Rule).all()


# =====================================================================
# Safe rule evaluation — NO eval()/exec()
# =====================================================================

_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "in": lambda a, b: a in b if isinstance(b, (list, tuple, set)) else False,
    "not_in": lambda a, b: a not in b if isinstance(b, (list, tuple, set)) else True,
    "contains": lambda a, b: str(b) in str(a),
    "starts_with": lambda a, b: str(a).startswith(str(b)),
}

_FIELD_RESOLVERS: Dict[str, Callable] = {
    "workflow.state": lambda wf: wf.state if hasattr(wf, "state") else None,
    "workflow.name": lambda wf: wf.name if hasattr(wf, "name") else None,
    "workflow.description": lambda wf: wf.description if hasattr(wf, "description") else None,
}

_VALID_WORKFLOW_STATES = {s.value for s in WorkflowState}


def _resolve_field(workflow: Any, field: str) -> Any:
    """Resolve a whitelisted field path to its value."""
    resolver = _FIELD_RESOLVERS.get(field)
    if resolver is None:
        logger.warning("Rule references unknown field: %s", field)
        return None
    return resolver(workflow)


def _evaluate_condition(condition: Any, workflow: Any) -> bool:
    """Evaluate a single condition or compound condition against a workflow."""
    if not isinstance(condition, dict):
        logger.warning("Rule condition is not a dict, skipping")
        return False

    # Compound: {"all": [...]}
    if "all" in condition:
        subconds = condition["all"]
        if not isinstance(subconds, list):
            return False
        return all(_evaluate_condition(c, workflow) for c in subconds)

    # Compound: {"any": [...]}
    if "any" in condition:
        subconds = condition["any"]
        if not isinstance(subconds, list):
            return False
        return any(_evaluate_condition(c, workflow) for c in subconds)

    # Simple: {"field": ..., "op": ..., "value": ...}
    field = condition.get("field")
    op = condition.get("op")
    expected = condition.get("value")

    if not field or not op:
        logger.warning("Rule condition missing field or op: %s", condition)
        return False

    if op not in _OPERATORS:
        logger.warning("Rule uses unknown operator: %s", op)
        return False

    actual = _resolve_field(workflow, field)
    try:
        return _OPERATORS[op](actual, expected)
    except (TypeError, ValueError):
        return False


def _execute_action(action: Any, workflow: Any) -> None:
    """Execute a safe, whitelisted action against a workflow."""
    if not isinstance(action, dict):
        logger.warning("Rule action is not a dict, skipping")
        return

    action_type = action.get("type")

    if action_type == "set_workflow_state":
        new_state = action.get("value")
        if new_state not in _VALID_WORKFLOW_STATES:
            logger.warning("Rule action has invalid state value: %s", new_state)
            return
        if hasattr(workflow, "state"):
            workflow.state = new_state
    else:
        logger.warning("Rule action has unknown type: %s", action_type)


def _parse_json_field(raw: Any) -> Optional[Any]:
    """Parse a rule field that may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def evaluate_rules_for_workflow(db: Session, workflow) -> None:
    """
    Evaluate all rules against a given workflow using the safe JSON DSL.

    This mutates the workflow/tasks in-memory. The caller is responsible for committing.
    """
    rules = list_rules(db)
    if not rules:
        return

    for rule in rules:
        try:
            condition = _parse_json_field(rule.condition)
            if condition is None:
                logger.warning(
                    "Rule %s (%s) has unparseable condition, skipping",
                    rule.id, rule.name,
                )
                continue

            action = _parse_json_field(rule.action)
            if action is None:
                logger.warning(
                    "Rule %s (%s) has unparseable action, skipping",
                    rule.id, rule.name,
                )
                continue

            if _evaluate_condition(condition, workflow):
                _execute_action(action, workflow)
        except Exception:
            logger.exception("Rule %s (%s) evaluation failed", rule.id, rule.name)
            continue
