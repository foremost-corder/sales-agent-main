"""One strict JSON protocol for every model-generated application payload."""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from pydantic import BaseModel


class StrictJsonSchemaError(ValueError):
    """Raised locally before an incompatible schema reaches a model provider."""


def strict_model_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a strict Structured Outputs schema from a Pydantic model.

    Pydantic omits fields with defaults from ``required``. Structured Outputs
    requires every property to be required, so defaults are request-builder
    conveniences only; the model must still emit every field.
    """

    schema = deepcopy(model.model_json_schema())
    _normalize_strict_schema(schema)
    _validate_strict_schema(schema)
    properties = schema.get("properties")
    version = properties.get("schema_version") if isinstance(properties, dict) else None
    if not isinstance(version, dict) or "const" not in version:
        raise StrictJsonSchemaError(
            f"{model.__name__} must define a literal top-level schema_version"
        )
    return schema


def strict_response_format(
    model: type[BaseModel], operation: str
) -> dict[str, Any]:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", operation).strip("_")[:64]
    if not name:
        raise StrictJsonSchemaError("response format name cannot be empty")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": strict_model_json_schema(model),
        },
    }


def strict_tool_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize function arguments to the same strict JSON rules."""

    normalized = deepcopy(schema)
    _normalize_strict_schema(normalized)
    _validate_strict_schema(normalized)
    return normalized


def strict_json_instruction(instruction: str) -> str:
    return (
        f"{instruction}\n"
        "响应由 API 的严格 JSON Schema（json_schema）约束。只返回该 JSON 对象，不要输出 Markdown、"
        "代码围栏、解释文字或额外字段；所有字段都必须出现。"
    )


def completion_content(response: Any) -> str | None:
    """Read standard and common OpenAI-compatible completion shapes."""

    if isinstance(response, str):
        value = response.strip()
        if not value:
            return None
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(decoded, str):
            return decoded.strip() or None
        if isinstance(decoded, dict) and "choices" in decoded:
            return _content_from_mapping(decoded)
        if isinstance(decoded, dict):
            return json.dumps(decoded, ensure_ascii=False)
        return value
    if isinstance(response, dict):
        return _content_from_mapping(response)
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return content if isinstance(content, str) else None


def _content_from_mapping(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _normalize_strict_schema(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_strict_schema(item)
        return
    if not isinstance(node, dict):
        return
    node.pop("default", None)
    properties = node.get("properties")
    if isinstance(properties, dict):
        additional = node.get("additionalProperties")
        if additional not in (None, False):
            raise StrictJsonSchemaError(
                "free-form object properties are not supported by the strict model protocol"
            )
        node["required"] = list(properties)
        node["additionalProperties"] = False
    for value in node.values():
        _normalize_strict_schema(value)


def _validate_strict_schema(node: Any, path: str = "$") -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _validate_strict_schema(item, f"{path}[{index}]")
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        required = node.get("required")
        if not isinstance(required, list) or required != list(properties):
            raise StrictJsonSchemaError(
                f"strict schema at {path} must require every property in declaration order"
            )
        if node.get("additionalProperties") is not False:
            raise StrictJsonSchemaError(
                f"strict schema at {path} must set additionalProperties=false"
            )
    for key, value in node.items():
        _validate_strict_schema(value, f"{path}.{key}")
