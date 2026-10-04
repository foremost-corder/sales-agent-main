from typing import Any, cast

import pytest

from sales_agent.agent.tool_runtime import EvidenceKind
from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult
from sales_agent.tools.registry import DuplicateToolNameError, ToolRegistry


class FakeProvider:
    def __init__(self, name: str, evidence_kind: str = "none") -> None:
        self.name = name
        self.evidence_kind = cast(EvidenceKind, evidence_kind)

    def tool_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=self.name,
                description=f"Execute {self.name}.",
                parameters={"type": "object", "properties": {}},
                evidence_kind=self.evidence_kind,
            )
        ]

    def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            ok=True,
            result={"arguments": arguments},
        )


def test_registry_composes_arbitrary_providers_and_propagates_evidence_metadata() -> None:
    registry = ToolRegistry(
        [
            FakeProvider("first_tool", "metadata"),
            FakeProvider("second_tool", "substantive"),
        ]
    )

    invocation = registry.invoke(
        tool_name="second_tool", arguments={"value": 42}
    )

    assert [item.name for item in registry.definitions()] == [
        "first_tool",
        "second_tool",
    ]
    assert invocation.succeeded is True
    assert invocation.arguments == {"value": 42}
    assert invocation.evidence_kind == "substantive"


def test_registry_rejects_duplicate_names_across_independent_providers() -> None:
    with pytest.raises(DuplicateToolNameError, match="duplicate tool name"):
        ToolRegistry([FakeProvider("same_tool"), FakeProvider("same_tool")])
