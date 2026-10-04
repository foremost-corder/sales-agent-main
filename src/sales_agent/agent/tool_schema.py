"""Validation for model-facing function-tool definitions."""

import re
from copy import deepcopy

from sales_agent.agent.structured_json import strict_tool_parameters
from sales_agent.agent.tool_runtime import FUNCTION_NAME_PATTERN, ToolSpec


def validate_chat_completions_tools(tools: list[dict[str, object]]) -> None:
    """Fail locally when a tool payload cannot satisfy Chat Completions."""
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"tools[{index}].type must be 'function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not re.fullmatch(FUNCTION_NAME_PATTERN, name):
            raise ValueError(
                f"tools[{index}].function.name must match {FUNCTION_NAME_PATTERN}"
            )
        description = function.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"tools[{index}].function.description must be a non-empty string"
            )
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(
                f"tools[{index}].function.parameters must be an object JSON Schema"
            )


def strict_chat_completions_tools(
    tools: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return strict copies so every model-generated argument is schema-bound."""

    validate_chat_completions_tools(tools)
    normalized = deepcopy(tools)
    for tool in normalized:
        function = tool["function"]
        function["parameters"] = strict_tool_parameters(function["parameters"])
        function["strict"] = True
    return normalized


def render_chat_completions_tools(tools: list[ToolSpec]) -> list[dict[str, object]]:
    """Translate neutral tool specs only at the OpenAI transport boundary."""

    payloads: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]
    return strict_chat_completions_tools(payloads)
