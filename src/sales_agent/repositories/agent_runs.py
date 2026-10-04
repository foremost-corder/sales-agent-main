"""Append-only audit persistence for agent executions and tool calls."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from sales_agent.domain.models import AgentRun, ToolCall


def create_agent_run(session: Session, *, conversation_id: uuid.UUID) -> AgentRun:
    run = AgentRun(conversation_id=conversation_id, status="running")
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def append_tool_call(session: Session, *, run_id: uuid.UUID, tool_name: str, arguments: dict[str, Any], result: dict[str, Any], succeeded: bool) -> ToolCall:
    call = ToolCall(run_id=run_id, tool_name=tool_name, arguments_json=arguments, result_json=result, status="completed" if succeeded else "failed", finished_at=datetime.now(timezone.utc))
    session.add(call)
    session.commit()
    session.refresh(call)
    return call


def finish_agent_run(session: Session, *, run: AgentRun, status: str, graph_state: dict[str, Any]) -> None:
    run.status = status
    run.graph_state = graph_state
    run.finished_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
