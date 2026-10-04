"""Stable names and validation for isolated knowledge corpora."""

import re


BASELINE_KNOWLEDGE_NAMESPACE = "baseline-react-v4"
LEAN_KNOWLEDGE_NAMESPACE = "experiment-lean-v1"
KNOWLEDGE_NAMESPACE_PATTERN = r"^[a-z][a-z0-9._-]{0,63}$"


def validate_knowledge_namespace(value: str) -> str:
    if re.fullmatch(KNOWLEDGE_NAMESPACE_PATTERN, value) is None:
        raise ValueError(
            "knowledge_namespace must start with a lowercase letter and contain "
            "only lowercase letters, digits, dots, underscores, or hyphens"
        )
    return value
