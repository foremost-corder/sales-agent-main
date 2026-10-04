"""Ports through which the core agent reaches an arbitrary tool catalog."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol


EvidenceKind = Literal["none", "metadata", "substantive"]
FUNCTION_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral description exposed to the model adapter."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolInvocation:
    """Transport-neutral outcome returned to the core ReAct loop."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    succeeded: bool
    evidence_kind: EvidenceKind


class ToolRuntime(Protocol):
    """Runtime port implemented by a registry, plugin host, or test double.

    ``definitions`` is intentionally evaluated before every model turn, so a
    large catalog runtime may expose a context-dependent subset without any
    change to the ReAct core.
    """

    def definitions(self) -> list[ToolSpec]: ...

    def invoke(
        self, *, tool_name: str, arguments: dict[str, Any]
    ) -> ToolInvocation: ...
