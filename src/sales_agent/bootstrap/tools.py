"""Compose built-in and externally supplied tools outside the ReAct core."""

from collections.abc import Iterable

from sqlalchemy.orm import Session

from sales_agent.features.knowledge.standard_tool import CallKnowledgeToolExecutor
from sales_agent.features.calls.tool import CallSemanticToolExecutor
from sales_agent.tools.contracts import ToolProvider
from sales_agent.features.knowledge.read_tools import KnowledgeReadToolExecutor
from sales_agent.features.knowledge.lean_tool import LeanCallKnowledgeToolExecutor
from sales_agent.features.single_call_scoring.tool import SingleCallWhiteboardToolExecutor
from sales_agent.tools.registry import ToolRegistry


def get_external_tool_providers() -> tuple[ToolProvider, ...]:
    """Dependency hook for configured MCP, plugin, or application providers."""
    return ()


def build_tool_runtime(
    session: Session,
    *,
    user_id: str,
    external_providers: Iterable[ToolProvider] = (),
) -> ToolRegistry:
    providers: list[ToolProvider] = [
        CallSemanticToolExecutor(session, user_id=user_id),
        CallKnowledgeToolExecutor(session, user_id=user_id),
        LeanCallKnowledgeToolExecutor(session, user_id=user_id),
        KnowledgeReadToolExecutor(session, user_id=user_id),
        SingleCallWhiteboardToolExecutor(session, user_id=user_id),
        *external_providers,
    ]
    return ToolRegistry(providers)
