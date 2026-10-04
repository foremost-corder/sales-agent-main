"""Persistence-neutral audit port for agent runs."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID


class RunAuditSink(Protocol):
    def start_run(self, *, conversation_id: UUID) -> str: ...

    def record_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        succeeded: bool,
    ) -> None: ...

    def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        graph_state: dict[str, Any],
    ) -> None: ...


class NullRunAuditSink:
    """Default adapter for pure runs that do not require persistence."""

    def start_run(self, *, conversation_id: UUID) -> str:
        return str(conversation_id)

    def record_tool_call(self, **_: Any) -> None:
        return None

    def finish_run(self, **_: Any) -> None:
        return None
