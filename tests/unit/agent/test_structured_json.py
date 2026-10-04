from typing import Any

import pytest

from sales_agent.agent.contracts import FinalAnswer
from sales_agent.agent.structured_json import (
    StrictJsonSchemaError,
    strict_model_json_schema,
    strict_tool_parameters,
)
from sales_agent.features.knowledge.contracts import SpeakerResolutionResult
from sales_agent.features.knowledge.evaluation import _EvaluationResult
from sales_agent.features.knowledge.fact_extractor import (
    _AnnotationResult,
    _CoverageResult,
    _RecallResult,
)
from sales_agent.features.knowledge.lean_fact_extractor import (
    _LeanExtractionOutput,
    _LeanReviewOutput,
)
from sales_agent.features.scoring.contracts import ReviewReport
from sales_agent.features.scoring.model_agents import _SectionOutput
from sales_agent.features.single_call_scoring.contracts import (
    CallLevelClassification,
    EvidenceReviewOutput,
    ModuleJudgmentOutput,
)


MODEL_OUTPUTS = [
    FinalAnswer,
    SpeakerResolutionResult,
    _EvaluationResult,
    _RecallResult,
    _AnnotationResult,
    _CoverageResult,
    _LeanExtractionOutput,
    _LeanReviewOutput,
    _SectionOutput,
    ReviewReport,
    CallLevelClassification,
    ModuleJudgmentOutput,
    EvidenceReviewOutput,
]


@pytest.mark.parametrize("model", MODEL_OUTPUTS)
def test_every_model_output_has_one_strict_versioned_schema(model) -> None:
    schema = strict_model_json_schema(model)

    assert schema["properties"]["schema_version"]["const"]
    _assert_every_object_is_closed_and_required(schema)
    _assert_keyword_absent(schema, "default")


def test_strict_tool_schema_rejects_free_form_objects_before_provider_request() -> None:
    with pytest.raises(StrictJsonSchemaError, match="free-form"):
        strict_tool_parameters(
            {
                "type": "object",
                "properties": {"metadata": {"type": "object"}},
                "additionalProperties": {"type": "string"},
            }
        )


def _assert_every_object_is_closed_and_required(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_every_object_is_closed_and_required(item)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        assert node["required"] == list(properties)
        assert node["additionalProperties"] is False
    for value in node.values():
        _assert_every_object_is_closed_and_required(value)


def _assert_keyword_absent(node: Any, keyword: str) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_keyword_absent(item, keyword)
        return
    if not isinstance(node, dict):
        return
    assert keyword not in node
    for value in node.values():
        _assert_keyword_absent(value, keyword)
