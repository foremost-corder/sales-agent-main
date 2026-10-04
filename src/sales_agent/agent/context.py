"""Model-message assembly for one bounded agent run."""

from typing import Any

from sales_agent.agent.contracts import ChatMessage


def build_run_context(
    messages: list[ChatMessage], *, summary: str | None
) -> list[dict[str, Any]]:
    initial_messages: list[dict[str, Any]] = []
    if summary:
        initial_messages.append(
            {
                "role": "developer",
                "content": f"以下是经过系统验证的历史摘要，仅用于补充上下文：\n{summary}",
            }
        )
    initial_messages.extend({"role": item.role, "content": item.content} for item in messages)
    return initial_messages
