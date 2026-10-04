"""SQLAlchemy implementation of the agent audit port."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from sales_agent.domain.models import AgentRun
from sales_agent.repositories.agent_runs import (
    append_tool_call,
    create_agent_run,
    finish_agent_run,
)


class SqlAlchemyRunAuditSink:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._runs: dict[str, AgentRun] = {}

    def start_run(self, *, conversation_id: UUID) -> str:
        run = create_agent_run(self._session, conversation_id=conversation_id)
        run_id = str(run.id)
        self._runs[run_id] = run
        return run_id

    def record_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        succeeded: bool,
    ) -> None:
        self._require_run(run_id)
        append_tool_call(
            self._session,
            run_id=UUID(run_id),
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            succeeded=succeeded,
        )

    def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        graph_state: dict[str, Any],
    ) -> None:
        run = self._require_run(run_id)
        finish_agent_run(
            self._session,
            run=run,
            status=status,
            graph_state=graph_state,
        )
        self._runs.pop(run_id, None)

    def _require_run(self, run_id: str) -> AgentRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown audit run: {run_id}") from exc
