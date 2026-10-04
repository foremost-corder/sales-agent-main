from types import SimpleNamespace

import pytest

from sales_agent.agent import chat_model as chat_model_module
from sales_agent.agent.chat_model import (
    ChatModelContractError,
    ChatModelNotConfiguredError,
    OpenAIChatCompletionsModel,
)
from sales_agent.core.config import Settings
from sales_agent.agent.tool_runtime import ToolSpec


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+psycopg://unused:unused@127.0.0.1/unused",
        "openai_api_key": "test-key",
        "chat_model": "test-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_chat_model_requires_api_key() -> None:
    model = OpenAIChatCompletionsModel(
        make_settings(openai_api_key="replace-with-your-api-key")
    )

    with pytest.raises(ChatModelNotConfiguredError):
        model.create_completion(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )


def test_chat_model_uses_chat_completions_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_options"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(chat_model_module, "OpenAI", FakeOpenAI)
    model = OpenAIChatCompletionsModel(make_settings())

    result = model.create_completion(
        messages=[{"role": "user", "content": "你好"}],
        tools=[
            ToolSpec(
                name="query_calls",
                description="Query calls.",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )

    assert result.choices == []
    assert captured["model"] == "test-model"
    assert captured["messages"] == [
        {"role": "system", "content": chat_model_module.SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": "你好"},
    ]
    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "query_calls",
                "description": "Query calls.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]
    assert captured["tool_choice"] == "auto"
    assert captured["parallel_tool_calls"] is False


def test_chat_model_rejects_malformed_tool_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("a malformed tool must not create an API client")

    monkeypatch.setattr(chat_model_module, "OpenAI", FakeOpenAI)
    model = OpenAIChatCompletionsModel(make_settings())

    with pytest.raises(ChatModelContractError, match="工具配置无效"):
        model.create_completion(
            messages=[{"role": "user", "content": "你好"}],
            tools=[
                ToolSpec(
                    name="query_calls",
                    description="Query calls.",
                    parameters={
                        "type": "object",
                        "properties": {"metadata": {"type": "object"}},
                        "additionalProperties": {"type": "string"},
                    },
                )
            ],
        )
