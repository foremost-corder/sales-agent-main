import json
from datetime import date

from sales_agent.agent.chat_model import ChatModelRequestError
from sales_agent.core.config import Settings
from sales_agent.features.scoring.contracts import CallTarget, ReviewReport, ScoreEvidence
from sales_agent.features.scoring.model_agents import (
    OpenAISectionScoringAgent,
    OpenAIScoreReviewAgent,
    _REVIEW_INSTRUCTION,
    _SECTION_INSTRUCTION,
    _SectionOutput,
    _completion_content,
    _json_system_instruction,
    _provider_error,
    _retry_after_seconds,
    _strict_response_format,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1


class ProviderError(Exception):
    def __init__(self, body):
        self.body = body


class UpstreamError(Exception):
    def __init__(self, status_code, *, headers=None):
        self.status_code = status_code
        self.response = type("Response", (), {"headers": headers or {}})()


def test_period_scoring_prompts_keep_business_calibration_lenient() -> None:
    for label in ("介绍业务", "承上启下", "挖到需求或信息", "介绍卖点/优势"):
        assert label in _SECTION_INSTRUCTION
        assert label in _REVIEW_INSTRUCTION
    assert "替代路径" in _REVIEW_INSTRUCTION
    assert "否定或状态信息" in _SECTION_INSTRUCTION


def test_provider_error_reads_sdk_unwrapped_body() -> None:
    assert _provider_error(
        ProviderError({"code": "context_length_exceeded", "message": "too long"})
    ) == ("context_length_exceeded", "too long")


def test_provider_error_reads_full_compatible_api_body() -> None:
    assert _provider_error(
        ProviderError({"error": {"code": "bad_request", "message": "invalid input"}})
    ) == ("bad_request", "invalid input")


def test_retry_after_reads_numeric_header() -> None:
    assert _retry_after_seconds(UpstreamError(429, headers={"retry-after": "2.5"})) == 2.5


def test_json_instruction_contains_gateway_compatible_lowercase_keyword() -> None:
    instruction = _json_system_instruction("只返回结构化结果。")

    assert "json" in instruction


def test_section_output_schema_is_compatible_with_strict_structured_outputs() -> None:
    response_format = _strict_response_format(_SectionOutput, "score section test")
    schema = response_format["json_schema"]["schema"]
    judgment = schema["$defs"]["RuleJudgment"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert set(judgment["required"]) == set(judgment["properties"])


def test_review_schema_is_compatible_with_strict_structured_outputs() -> None:
    response_format = _strict_response_format(ReviewReport, "score review")
    schema = response_format["json_schema"]["schema"]

    assert set(schema["required"]) == set(schema["properties"])
    issue = schema["$defs"]["ReviewIssue"]
    assert set(issue["required"]) == set(issue["properties"])


def test_review_timeout_returns_failed_report_instead_of_losing_scores(
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise ChatModelRequestError("score review request failed")

    monkeypatch.setattr("sales_agent.features.scoring.model_agents._complete", fail)
    reviewer = OpenAIScoreReviewAgent(
        Settings(
            _env_file=None,
            database_url="postgresql+psycopg://unused",
            openai_api_key="test-key",
        )
    )

    report = reviewer.review(policy=SALES_KEY_BEHAVIOR_POLICY_V1, calls=[])

    assert report.status == "failed"
    assert report.reasonable is False
    assert report.issues == []
    assert "结果已保留" in report.summary


def test_completion_content_accepts_direct_json_string() -> None:
    raw = '{"schema_version":"score-section-v1","judgments":[]}'

    assert json.loads(_completion_content(raw)) == json.loads(raw)


def test_completion_content_accepts_json_encoded_completion_envelope() -> None:
    raw = (
        '{"choices":[{"message":{"content":'
        '"{\\"schema_version\\":\\"score-section-v1\\",\\"judgments\\":[]}"}}]}'
    )

    assert _completion_content(raw) == (
        '{"schema_version":"score-section-v1","judgments":[]}'
    )


def test_completion_content_accepts_mapping_completion_envelope() -> None:
    response = {"choices": [{"message": {"content": "{\"ok\":true}"}}]}

    assert _completion_content(response) == '{"ok":true}'


def test_section_agent_sends_compact_model_payload(monkeypatch) -> None:
    captured = {}

    def complete(settings, instruction, payload, operation, schema):
        captured.update(payload)
        captured["schema"] = schema
        return '{"schema_version":"score-section-v1","judgments":[]}'

    monkeypatch.setattr("sales_agent.features.scoring.model_agents._complete", complete)
    section = SALES_KEY_BEHAVIOR_POLICY_V1.sections[0]
    rule = SALES_KEY_BEHAVIOR_POLICY_V1.rules[0]
    call = CallTarget(
        call_id="call-1",
        external_call_id="external-1",
        call_date=date(2026, 9, 10),
        sales_stage="销售线索",
        analysis_run_id="run-1",
        analysis_degraded=True,
    )
    evidence = ScoreEvidence(
        evidence_id="evidence-1",
        retrieved_for_rule_id=rule.rule_id,
        source="vector_fact",
        call_id=call.call_id,
        analysis_run_id="run-1",
        fact_id="fact-1",
        document_id="document-1",
        trigger_validation_status="verified",
        fact_type=rule.rule_id,
        speaker="sales",
        fact="销售介绍公司从事电脑租赁。",
        turn_no=3,
        quote="我们公司主要从事电脑租赁。",
        grounding="exact",
        confidence=0.99,
        similarity=0.88,
    )

    OpenAISectionScoringAgent(
        Settings(
            _env_file=None,
            database_url="postgresql+psycopg://unused",
            openai_api_key="test-key",
        )
    ).score_section(section=section, rules=[rule], calls=[call], evidence=[evidence])

    assert captured["calls"] == [{"call_id": "call-1"}]
    assert captured["schema"] is _SectionOutput
    assert "output_schema" not in captured
    assert set(captured["rules"][0]) == {
        "rule_id",
        "name",
        "explanation",
        "criteria",
        "points_per_match",
        "maximum_points",
        "scoring_mode",
    }
    assert set(captured["candidate_evidence"][0]) == {
        "evidence_id",
        "retrieved_for_rule_id",
        "source",
        "call_id",
        "speaker",
        "fact",
        "turn_no",
        "quote",
        "grounding",
    }


def test_transcript_context_omits_low_confidence_fact_summary(monkeypatch) -> None:
    captured = {}

    def complete(settings, instruction, payload, operation, schema):
        captured.update(payload)
        return '{"schema_version":"score-section-v1","judgments":[]}'

    monkeypatch.setattr("sales_agent.features.scoring.model_agents._complete", complete)
    section = SALES_KEY_BEHAVIOR_POLICY_V1.sections[0]
    rule = SALES_KEY_BEHAVIOR_POLICY_V1.rules[0]
    call = CallTarget(
        call_id="call-1",
        external_call_id="external-1",
        call_date=date(2026, 9, 10),
        sales_stage="销售线索",
    )
    evidence = ScoreEvidence(
        evidence_id="evidence-1",
        retrieved_for_rule_id=rule.rule_id,
        source="transcript_context",
        call_id=call.call_id,
        speaker="sales",
        fact="不应发送给评分模型的低置信度摘要",
        turn_no=3,
        quote="应该直接判断的通话原文",
        grounding="aligned",
        confidence=0.5,
    )

    OpenAISectionScoringAgent(
        Settings(
            _env_file=None,
            database_url="postgresql+psycopg://unused",
            openai_api_key="test-key",
        )
    ).score_section(section=section, rules=[rule], calls=[call], evidence=[evidence])

    sent = captured["candidate_evidence"][0]
    assert sent["quote"] == "应该直接判断的通话原文"
    assert "fact" not in sent


def test_section_agent_retries_transient_502_then_succeeds(monkeypatch) -> None:
    calls = 0
    sleeps = []

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise UpstreamError(502)
            return '{"schema_version":"score-section-v1","judgments":[]}'

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("sales_agent.features.scoring.model_agents.OpenAI", Client)
    monkeypatch.setattr("sales_agent.features.scoring.model_agents.random.uniform", lambda *_: 0)
    monkeypatch.setattr("sales_agent.features.scoring.model_agents.time.sleep", sleeps.append)
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused",
        openai_api_key="test-key",
        scoring_max_retries=3,
        scoring_retry_base_seconds=1,
    )

    result = OpenAISectionScoringAgent(settings).score_section(
        section=SALES_KEY_BEHAVIOR_POLICY_V1.sections[0],
        rules=[SALES_KEY_BEHAVIOR_POLICY_V1.rules[0]],
        calls=[],
        evidence=[],
    )

    assert result == []
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_section_agent_does_not_retry_bad_request(monkeypatch) -> None:
    calls = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise UpstreamError(400)

    class Client:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("sales_agent.features.scoring.model_agents.OpenAI", Client)
    monkeypatch.setattr(
        "sales_agent.features.scoring.model_agents.time.sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused",
        openai_api_key="test-key",
        scoring_max_retries=3,
    )

    try:
        OpenAISectionScoringAgent(settings).score_section(
            section=SALES_KEY_BEHAVIOR_POLICY_V1.sections[0],
            rules=[SALES_KEY_BEHAVIOR_POLICY_V1.rules[0]],
            calls=[],
            evidence=[],
        )
    except ChatModelRequestError as exc:
        assert exc.status_code == 400
        assert exc.attempts == 1
        assert "after 1 attempt(s)" in str(exc)
    else:
        raise AssertionError("expected ChatModelRequestError")

    assert calls == 1
