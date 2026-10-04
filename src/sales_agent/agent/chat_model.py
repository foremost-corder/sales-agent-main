"""OpenAI Chat Completions adapter, isolated from agent orchestration."""

from functools import lru_cache
from typing import Any

from openai import OpenAI

from sales_agent.agent.contracts import ChatModel
from sales_agent.agent.prompts import SYSTEM_INSTRUCTIONS
from sales_agent.agent.tool_schema import render_chat_completions_tools
from sales_agent.core.config import Settings, get_settings


class ChatModelNotConfiguredError(RuntimeError):
    pass


class ChatModelRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.provider_code = provider_code
        self.provider_message = provider_message
        self.attempts = attempts

    @property
    def public_detail(self) -> str:
        parts = ["模型服务请求失败"]
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.request_id:
            parts.append(f"请求 ID: {self.request_id}")
        if self.provider_code:
            parts.append(f"服务商错误: {self.provider_code}")
        if self.attempts is not None and self.attempts > 1:
            parts.append(f"已尝试 {self.attempts} 次")
        return f"{parts[0]}（{'；'.join(parts[1:])}）" if len(parts) > 1 else parts[0]


class ChatModelContractError(RuntimeError):
    """Raised before a malformed local tool payload can leave this process."""


class OpenAIChatCompletionsModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        if not self._settings.has_openai_api_key:
            raise ChatModelNotConfiguredError("OPENAI_API_KEY is not configured")
        try:
            strict_tools = render_chat_completions_tools(tools)
        except ValueError as exc:
            raise ChatModelContractError(f"Chat Completions 工具配置无效：{exc}") from exc
        client_options: dict[str, object] = {
            "api_key": self._settings.openai_api_key.get_secret_value(),
            "max_retries": 1,
            "timeout": 30.0,
        }
        if self._settings.openai_base_url:
            client_options["base_url"] = self._settings.openai_base_url
        client = OpenAI(**client_options)
        try:
            return client.chat.completions.create(
                model=self._settings.chat_model,
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTIONS}, *messages],
                tools=strict_tools,
                # Standard ReAct: the model may answer, call a tool, or explain
                # that no available tool can satisfy the request.
                tool_choice="auto",
                parallel_tool_calls=False,
                max_completion_tokens=self._settings.chat_max_output_tokens,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            request_id = getattr(exc, "request_id", None)
            raise ChatModelRequestError(
                "model request failed",
                status_code=status_code if isinstance(status_code, int) else None,
                request_id=request_id if isinstance(request_id, str) else None,
            ) from exc


@lru_cache
def get_chat_model() -> ChatModel:
    return OpenAIChatCompletionsModel(get_settings())
