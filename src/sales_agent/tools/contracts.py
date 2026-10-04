"""Shared, transport-neutral contracts for agent tools."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sales_agent.agent.tool_runtime import EvidenceKind, FUNCTION_NAME_PATTERN, ToolSpec


class ToolInput(BaseModel):
    """Base class for model-supplied tool arguments."""

    model_config = ConfigDict(extra="forbid")


class ToolError(BaseModel):
    code: str
    message: str


class ToolExecutionResult(BaseModel):
    """Stable envelope suitable for a tool message and for audit storage."""

    tool_name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: ToolError | None = None


class ToolDefinition(BaseModel):
    """Transport-neutral function schema for the agent tool registry."""

    name: str = Field(pattern=FUNCTION_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=1024)
    parameters: dict[str, Any]
    source: Literal["local", "mcp"] = "local"
    read_only: bool = True
    requires_approval: bool = False
    evidence_kind: EvidenceKind = "none"

    @field_validator("parameters")
    @classmethod
    def parameters_must_be_an_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("function parameters must be a JSON Schema object")
        return value

    def as_tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolProvider(Protocol):
    """One source of tools, whether implemented locally or exposed by MCP."""

    def tool_definitions(self) -> list[ToolDefinition]: ...

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult: ...
