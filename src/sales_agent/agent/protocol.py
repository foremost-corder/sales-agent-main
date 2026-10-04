"""Chat Completions message normalization for the agent loop."""

import json
from typing import Any

from sales_agent.agent.chat_model import ChatModelRequestError
from sales_agent.agent.contracts import ToolRequest


def tool_requests(response: Any) -> list[ToolRequest]:
    requests: list[ToolRequest] = []
    for tool_call in getattr(choice_message(response), "tool_calls", None) or []:
        function = tool_call.function
        requests.append(
            ToolRequest(
                call_id=tool_call.id,
                name=function.name,
                arguments=parse_arguments(function.arguments),
            )
        )
    return requests


def assistant_message(response: Any) -> dict[str, Any]:
    message = choice_message(response)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    normalized: dict[str, Any] = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        normalized["tool_calls"] = [
            {
                "id": call.id,
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    return normalized


def assistant_content(response: Any) -> str:
    message = choice_message(response)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()
    return ""


def tool_message(call_id: str, output: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(output, ensure_ascii=False),
    }


def parse_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def choice_message(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ChatModelRequestError("model response has no choices")
    return choices[0].message
