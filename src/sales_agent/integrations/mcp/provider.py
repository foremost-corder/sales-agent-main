"""Adapt allowlisted MCP tools to the application's common tool registry."""

from dataclasses import dataclass
import re
from typing import Any, Protocol

from sales_agent.tools.contracts import ToolDefinition, ToolExecutionResult


@dataclass(frozen=True)
class McpRemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


class McpClient(Protocol):
    """Transport adapter implemented with an MCP SDK or a test double."""

    def list_tools(self) -> list[McpRemoteTool]: ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class McpToolPolicy:
    """Fail-closed policy for one MCP server."""

    allowed_tools: frozenset[str]
    allow_write_tools: bool = False


class McpToolProvider:
    """Expose MCP tools as namespaced function tools with a strict allowlist.

    Write-capable tools remain blocked at execution time until the application
    supplies a real approval workflow. This prevents a prompt from bypassing
    authorization merely by selecting a tool.
    """

    def __init__(
        self,
        *,
        server_label: str,
        client: McpClient,
        policy: McpToolPolicy,
        user_id: str,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", server_label):
            raise ValueError("invalid MCP server label")
        self._client = client
        self._policy = policy
        self._user_id = user_id
        self._remote_by_exposed: dict[str, McpRemoteTool] = {}
        self._definitions: list[ToolDefinition] = []

        for remote in client.list_tools():
            if remote.name not in policy.allowed_tools:
                continue
            if not remote.read_only and not policy.allow_write_tools:
                continue
            exposed_name = f"mcp__{server_label}__{remote.name}"
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", exposed_name):
                raise ValueError(f"MCP tool name cannot be exposed safely: {remote.name}")
            requires_approval = not remote.read_only
            self._remote_by_exposed[exposed_name] = remote
            self._definitions.append(
                ToolDefinition(
                    name=exposed_name,
                    description=remote.description,
                    parameters=remote.input_schema,
                    source="mcp",
                    read_only=remote.read_only,
                    requires_approval=requires_approval,
                    evidence_kind="substantive" if remote.read_only else "none",
                )
            )

    def tool_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        remote = self._remote_by_exposed.get(tool_name)
        if remote is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "unknown_tool", "message": "MCP 工具未授权或不存在。"},
            )
        if not remote.read_only:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={
                    "code": "approval_required",
                    "message": "该 MCP 工具会修改外部数据，必须先进入用户批准流程。",
                },
            )
        try:
            result = self._client.call_tool(
                remote.name,
                arguments,
                user_id=self._user_id,
            )
        except Exception:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "mcp_call_failed", "message": "MCP 工具调用失败。"},
            )
        return ToolExecutionResult(tool_name=tool_name, ok=True, result=result)

    def execute_approved(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        """Execute a write tool only from a separate, audited approval node."""
        remote = self._remote_by_exposed.get(tool_name)
        if remote is None or remote.read_only or not self._policy.allow_write_tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "not_approvable", "message": "该 MCP 工具不可进入写操作批准流程。"},
            )
        try:
            result = self._client.call_tool(
                remote.name,
                arguments,
                user_id=self._user_id,
            )
        except Exception:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error={"code": "mcp_call_failed", "message": "MCP 工具调用失败。"},
            )
        return ToolExecutionResult(tool_name=tool_name, ok=True, result=result)
