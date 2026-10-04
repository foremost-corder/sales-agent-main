from typing import Any

from sales_agent.integrations.mcp.provider import (
    McpRemoteTool,
    McpToolPolicy,
    McpToolProvider,
)


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def list_tools(self) -> list[McpRemoteTool]:
        return [
            McpRemoteTool(
                name="find_customer",
                description="在 CRM 中查询客户。",
                input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                read_only=True,
            ),
            McpRemoteTool(
                name="update_customer",
                description="修改 CRM 客户。",
                input_schema={"type": "object", "properties": {}},
                read_only=False,
            ),
            McpRemoteTool(
                name="not_allowed",
                description="未授权。",
                input_schema={"type": "object", "properties": {}},
                read_only=True,
            ),
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]:
        self.calls.append((name, arguments, user_id))
        return {"customer_id": "c-1"}


def test_mcp_provider_namespaces_and_allowlists_tools() -> None:
    client = FakeMcpClient()
    provider = McpToolProvider(
        server_label="crm",
        client=client,
        policy=McpToolPolicy(
            allowed_tools=frozenset({"find_customer", "update_customer"})
        ),
        user_id="user-1",
    )

    definitions = provider.tool_definitions()

    assert [item.name for item in definitions] == ["mcp__crm__find_customer"]
    assert definitions[0].source == "mcp"
    assert definitions[0].read_only is True


def test_mcp_provider_executes_reads_and_blocks_writes() -> None:
    client = FakeMcpClient()
    provider = McpToolProvider(
        server_label="crm",
        client=client,
        policy=McpToolPolicy(
            allowed_tools=frozenset({"find_customer", "update_customer"}),
            allow_write_tools=True,
        ),
        user_id="user-1",
    )

    read = provider.execute("mcp__crm__find_customer", {"name": "Ada"})
    write = provider.execute("mcp__crm__update_customer", {})

    assert read.ok is True
    assert read.result == {"customer_id": "c-1"}
    assert client.calls == [("find_customer", {"name": "Ada"}, "user-1")]
    assert write.ok is False
    assert write.error is not None
    assert write.error.code == "approval_required"

    approved = provider.execute_approved("mcp__crm__update_customer", {})

    assert approved.ok is True
    assert client.calls[-1] == ("update_customer", {}, "user-1")
