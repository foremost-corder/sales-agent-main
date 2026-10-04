import pytest

from sales_agent.core.config import Settings
from sales_agent.features.knowledge.evaluation import (
    KnowledgeEvaluationError,
    OpenAIKnowledgeOutputEvaluator,
)


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused:unused@127.0.0.1/unused",
        openai_api_key="test-key",
        chat_model="test-model",
    )


def test_model_evaluator_returns_semantic_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OpenAIKnowledgeOutputEvaluator(make_settings())
    monkeypatch.setattr(
        evaluator,
        "_complete",
        lambda _, **__: """{
            "schema_version":"knowledge-evaluation-v1",
            "passed":false,
            "issues":[{"code":"missing_fact","path":["facts"],"message":"遗漏事实"}]
        }""",
    )

    issues = evaluator.evaluate(
        task_name="fact_extraction",
        source={"transcript": []},
        candidate={"facts": []},
        criteria=["不得遗漏事实"],
    )

    assert issues[0].source == "semantic"
    assert issues[0].code == "missing_fact"


def test_model_evaluator_rejects_inconsistent_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OpenAIKnowledgeOutputEvaluator(make_settings())
    monkeypatch.setattr(
        evaluator,
        "_complete",
        lambda _, **__: """{
            "schema_version":"knowledge-evaluation-v1",
            "passed":true,
            "issues":[{"code":"missing_fact","path":[],"message":"遗漏事实"}]
        }""",
    )

    with pytest.raises(KnowledgeEvaluationError):
        evaluator.evaluate(
            task_name="fact_extraction",
            source={},
            candidate={},
            criteria=[],
        )


def test_model_evaluator_repairs_its_own_schema_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = OpenAIKnowledgeOutputEvaluator(make_settings(), max_repairs=3)
    outputs = [
        '{"ok":true}',
        '{"schema_version":"knowledge-evaluation-v1","passed":true,"issues":[]}',
    ]
    requests: list[dict[str, object]] = []

    def fake_complete(payload: dict[str, object], *, repairing: bool) -> str:
        requests.append({"payload": payload, "repairing": repairing})
        return outputs.pop(0)

    monkeypatch.setattr(evaluator, "_complete", fake_complete)
    issues = evaluator.evaluate(
        task_name="fact_extraction",
        source={},
        candidate={},
        criteria=[],
    )

    assert issues == []
    repair_context = requests[1]["payload"]["repair_context"]
    assert repair_context["validation_errors"]
    assert requests[1]["repairing"] is True
