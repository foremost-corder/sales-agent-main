"""Generic name-based routing across an arbitrary set of tool providers."""

from collections.abc import Iterable
from typing import Any

from sales_agent.agent.tool_runtime import ToolInvocation
from sales_agent.agent.tool_runtime import ToolSpec
from sales_agent.tools.contracts import (
    ToolDefinition,
    ToolExecutionResult,
    ToolProvider,
)


class DuplicateToolNameError(ValueError):
    pass


class ToolRegistry:
    """Infrastructure adapter; it has no knowledge of concrete business tools."""

    def __init__(self, providers: Iterable[ToolProvider] = ()) -> None:
        self._definitions: list[ToolDefinition] = []
        self._routes: dict[str, ToolProvider] = {}
        self._by_name: dict[str, ToolDefinition] = {}
        for provider in providers:
            for definition in provider.tool_definitions():
                if definition.name in self._routes:
                    raise DuplicateToolNameError(
                        f"duplicate tool name: {definition.name}"
                    )
                self._definitions.append(definition)
                self._routes[definition.name] = provider
                self._by_name[definition.name] = definition

    def definitions(self) -> list[ToolSpec]:
        return [item.as_tool_spec() for item in self._definitions]

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "source": item.source,
                "read_only": item.read_only,
                "requires_approval": item.requires_approval,
                "evidence_kind": item.evidence_kind,
            }
            for item in self._definitions
        ]

    def invoke(
        self, *, tool_name: str, arguments: dict[str, Any]
    ) -> ToolInvocation:
        definition = self._by_name.get(tool_name)
        provider = self._routes.get(tool_name)
        if provider is None:
            result = ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={
                    "code": "unknown_tool",
                    "message": "请求的工具不在允许列表中。",
                },
            )
            return _invocation(definition, arguments, result)
        return _invocation(
            definition,
            arguments,
            provider.execute(tool_name, arguments),
        )

def _invocation(
    definition: ToolDefinition | None,
    arguments: dict[str, Any],
    result: ToolExecutionResult,
) -> ToolInvocation:
    return ToolInvocation(
        name=result.tool_name,
        arguments=arguments,
        output=result.model_dump(mode="json"),
        succeeded=result.ok,
        evidence_kind=definition.evidence_kind if definition else "none",
    )
