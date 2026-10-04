import pytest

from sales_agent.agent.tool_runtime import ToolSpec
from sales_agent.agent.tool_schema import validate_chat_completions_tools
from sales_agent.tools.contracts import ToolDefinition


def test_function_definition_uses_chat_completions_shape() -> None:
    definition = ToolDefinition(
        name="query_calls",
        description="Query calls.",
        parameters={"type": "object", "properties": {}},
    )

    assert definition.as_tool_spec() == ToolSpec(
        name="query_calls",
        description="Query calls.",
        parameters={"type": "object", "properties": {}},
    )


def test_tool_definition_rejects_an_empty_function_name() -> None:
    with pytest.raises(ValueError, match="String should match pattern"):
        ToolDefinition(name="", description="Query calls.", parameters={"type": "object"})


def test_chat_completions_tool_validation_rejects_responses_shape() -> None:
    with pytest.raises(ValueError, match=r"tools\[0\]\.function must be an object"):
        validate_chat_completions_tools([{"type": "function", "name": "query_calls"}])


def test_chat_completions_tool_validation_rejects_a_blank_name() -> None:
    with pytest.raises(ValueError, match=r"tools\[0\]\.function.name"):
        validate_chat_completions_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "",
                        "description": "Query calls.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
