"""Shared reliable transport for stateless strict-JSON model calls."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.agent.structured_json import completion_content, strict_json_instruction, strict_response_format
from sales_agent.core.config import Settings


logger = logging.getLogger(__name__)


class StrictJsonCompletionError(ValueError):
    pass


def complete_strict_json(
    settings: Settings,
    *,
    instruction: str,
    payload: dict[str, Any],
    operation: str,
    schema: type[BaseModel],
) -> str:
    if not settings.has_openai_api_key:
        raise ChatModelNotConfiguredError("OPENAI_API_KEY is not configured")
    assert settings.openai_api_key is not None
    options: dict[str, object] = {
        "api_key": settings.openai_api_key.get_secret_value(),
        "max_retries": 0,
        "timeout": settings.scoring_timeout_seconds,
    }
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url
    serialized = json.dumps(payload, ensure_ascii=False)
    client = OpenAI(**options)
    attempts = settings.scoring_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=settings.scoring_model_name,
                messages=[
                    {"role": "system", "content": strict_json_instruction(instruction)},
                    {"role": "user", "content": serialized},
                ],
                response_format=strict_response_format(schema, operation),
                max_completion_tokens=settings.scoring_max_output_tokens,
            )
            content = completion_content(response)
            if not content:
                raise StrictJsonCompletionError(f"{operation} returned empty content")
            return content
        except StrictJsonCompletionError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            provider_code, provider_message = _provider_error(exc)
            retryable = status in {408, 409, 425, 429} or (
                isinstance(status, int) and 500 <= status <= 599
            ) or isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {
                "APIConnectionError", "APITimeoutError", "ConnectError",
                "ConnectTimeout", "ReadError", "ReadTimeout",
            }
            if retryable and attempt < attempts:
                retry_after = _retry_after_seconds(exc)
                delay = min(retry_after, settings.scoring_retry_max_seconds) if retry_after is not None else min(
                    settings.scoring_retry_base_seconds * (2 ** (attempt - 1))
                    * (1 + random.uniform(0, 0.25)), settings.scoring_retry_max_seconds,
                )
                logger.warning(
                    "strict_json_retry operation=%s attempt=%s max_attempts=%s status=%s delay=%.3f",
                    operation, attempt, attempts, status, delay,
                )
                time.sleep(delay)
                continue
            raise ChatModelRequestError(
                f"{operation} request failed after {attempt} attempt(s)",
                status_code=status if isinstance(status, int) else None,
                request_id=(getattr(exc, "request_id", None)
                            if isinstance(getattr(exc, "request_id", None), str) else None),
                provider_code=provider_code,
                provider_message=provider_message,
                attempts=attempt,
            ) from exc
    raise StrictJsonCompletionError(f"{operation} did not complete")


def _provider_error(exc: Exception) -> tuple[str | None, str | None]:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) and isinstance(body, dict):
        error = body
    if not isinstance(error, dict):
        return None, None
    code = error.get("code")
    message = error.get("message")
    return (
        str(code)[:128] if code is not None else None,
        str(message)[:1000] if message is not None else None,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None
