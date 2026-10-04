"""Model adapters for section scoring and holistic score review."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.agent.structured_json import (
    StrictJsonSchemaError,
    _validate_strict_schema,
    completion_content as _completion_content,
    strict_json_instruction as _json_system_instruction,
    strict_response_format,
)
from sales_agent.core.config import Settings
from sales_agent.features.scoring.contracts import (
    CallScore,
    CallTarget,
    ReviewIssue,
    ReviewReport,
    RuleJudgment,
    ScoreEvidence,
    ScoreRule,
    ScoreSection,
    ScoringPolicy,
)


logger = logging.getLogger(__name__)


class ScoringAgentError(ValueError):
    pass


class _SectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["score-section-v1"]
    judgments: list[RuleJudgment] = Field(max_length=1000)


class OpenAISectionScoringAgent:
    """Judge one bounded section without retaining conversation state."""

    version = "score-section-v1"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def score_section(
        self,
        *,
        section: ScoreSection,
        rules: list[ScoreRule],
        calls: list[CallTarget],
        evidence: list[ScoreEvidence],
        repair_issues: list[ReviewIssue] = (),
    ) -> list[RuleJudgment]:
        payload: dict[str, Any] = {
            "section": {
                "section_id": section.section_id,
                "name": section.name,
                "purpose": section.purpose,
            },
            "rules": [_score_rule_payload(item) for item in rules],
            "calls": [{"call_id": item.call_id} for item in calls],
            "candidate_evidence": [_score_evidence_payload(item) for item in evidence],
            "review_issues": [item.model_dump(mode="json") for item in repair_issues],
        }
        operation = f"score section {section.section_id}"
        raw = _complete(
            self._settings,
            _SECTION_INSTRUCTION,
            payload,
            operation,
            _SectionOutput,
        )
        try:
            return _SectionOutput.model_validate_json(raw).judgments
        except ValidationError as exc:
            _log_validation_failure(operation, exc)
            raise ScoringAgentError(
                f"{operation} violated the strict response schema"
            ) from exc


def _score_rule_payload(rule: ScoreRule) -> dict[str, Any]:
    """Keep scoring semantics while excluding retrieval-only configuration."""
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "explanation": rule.explanation,
        "criteria": rule.criteria,
        "points_per_match": rule.points_per_match,
        "maximum_points": rule.maximum_points,
        "scoring_mode": rule.scoring_mode,
    }


def _score_evidence_payload(evidence: ScoreEvidence) -> dict[str, Any]:
    """Serialize only fields that can affect a model's evidence judgment."""
    payload: dict[str, Any] = {
        "evidence_id": evidence.evidence_id,
        "retrieved_for_rule_id": evidence.retrieved_for_rule_id,
        "source": evidence.source,
        "call_id": evidence.call_id,
        "speaker": evidence.speaker,
        "quote": evidence.quote,
        "grounding": evidence.grounding,
    }
    if evidence.turn_no is not None:
        payload["turn_no"] = evidence.turn_no
    if evidence.source != "transcript_context":
        payload["fact"] = evidence.fact
    return payload


class OpenAIScoreReviewAgent:
    """Review one structured score batch without retaining conversation state."""

    version = "score-review-v1"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def review(
        self,
        *,
        policy: ScoringPolicy,
        calls: list[CallScore],
    ) -> ReviewReport:
        scored_calls = []
        for call in calls:
            scored_rules = [
                {
                    "rule_id": rule.rule_id,
                    "points": rule.points,
                    "matched_count": rule.matched_count,
                    "rationale": rule.rationale[:300],
                    "evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "source": item.source,
                            "quote": item.quote[:240],
                        }
                        for item in rule.evidence
                    ],
                }
                for rule in call.rule_scores
                if rule.points != 0
            ]
            if scored_rules:
                scored_calls.append(
                    {
                        "call_id": call.call_id,
                        "external_call_id": call.external_call_id,
                        "call_date": call.call_date.isoformat(),
                        "score": call.score,
                        "scored_rules": scored_rules,
                    }
                )
        payload = {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "criteria": rule.criteria,
                    "points_per_match": rule.points_per_match,
                    "maximum_points": rule.maximum_points,
                    "scoring_mode": rule.scoring_mode,
                }
                for rule in policy.rules
            ],
            "period_call_count": len(calls),
            "scored_calls": scored_calls,
        }
        try:
            raw = _complete(
                self._settings,
                _REVIEW_INSTRUCTION,
                payload,
                "score review",
                ReviewReport,
            )
        except ChatModelRequestError as exc:
            return ReviewReport(
                status="failed",
                reasonable=False,
                issues=[],
                summary=f"评分结果已保留，但模型审核失败：{exc.public_detail}",
            )
        try:
            return ReviewReport.model_validate_json(raw)
        except ValidationError as exc:
            _log_validation_failure("score review", exc)
            return ReviewReport(
                status="failed",
                reasonable=False,
                issues=[],
                summary="评分结果已保留，但模型审核违反严格响应结构。",
            )


def _complete(
    settings: Settings,
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
        # Retry here so attempts, delays, and the final upstream error remain
        # observable. Leaving SDK retries enabled would multiply attempts.
        "max_retries": 0,
        "timeout": settings.scoring_timeout_seconds,
    }
    if settings.openai_base_url:
        options["base_url"] = settings.openai_base_url
    serialized_payload = json.dumps(payload, ensure_ascii=False)
    # Some OpenAI-compatible gateways validate this requirement with a
    # case-sensitive literal search even when the prompt already says "JSON".
    json_instruction = _json_system_instruction(instruction)
    logger.info(
        "scoring_model_request operation=%s model=%s input_characters=%s",
        operation,
        settings.scoring_model_name,
        len(serialized_payload),
    )
    response_format = _strict_response_format(schema, operation)
    client = OpenAI(**options)
    total_attempts = settings.scoring_max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=settings.scoring_model_name,
                messages=[
                    {"role": "system", "content": json_instruction},
                    {"role": "user", "content": serialized_payload},
                ],
                response_format=response_format,
                max_completion_tokens=settings.scoring_max_output_tokens,
            )
            break
        except Exception as exc:
            status_code = _status_code(exc)
            request_id = getattr(exc, "request_id", None)
            provider_code, provider_message = _provider_error(exc)
            retryable = _is_retryable_model_error(exc, status_code)
            if retryable and attempt < total_attempts:
                delay = _retry_delay_seconds(exc, settings, attempt)
                # Deliberately do not log payload, messages, transcripts, or evidence.
                logger.warning(
                    "scoring_model_request_retry operation=%s model=%s "
                    "attempt=%s max_attempts=%s delay_seconds=%.3f "
                    "status_code=%s request_id=%s provider_code=%s error_type=%s",
                    operation,
                    settings.scoring_model_name,
                    attempt,
                    total_attempts,
                    delay,
                    status_code,
                    request_id,
                    provider_code,
                    type(exc).__name__,
                )
                time.sleep(delay)
                continue
            logger.error(
                "scoring_model_request_failed operation=%s model=%s "
                "attempts=%s retryable=%s status_code=%s request_id=%s "
                "provider_code=%s provider_message=%s error_type=%s",
                operation,
                settings.scoring_model_name,
                attempt,
                retryable,
                status_code,
                request_id,
                provider_code,
                provider_message,
                type(exc).__name__,
            )
            raise ChatModelRequestError(
                f"{operation} request failed after {attempt} attempt(s)",
                status_code=status_code,
                request_id=request_id if isinstance(request_id, str) else None,
                provider_code=provider_code,
                provider_message=provider_message,
                attempts=attempt,
            ) from exc
    content = _completion_content(response)
    if not isinstance(content, str) or not content.strip():
        raise ScoringAgentError(f"{operation} returned empty content")
    return content


def _log_validation_failure(operation: str, exc: ValidationError) -> None:
    logger.error(
        "scoring_model_schema_violation operation=%s validation_errors=%s",
        operation,
        json.dumps(
            exc.errors(include_url=False, include_input=False),
            ensure_ascii=False,
        ),
    )


def _strict_response_format(
    schema: type[BaseModel], operation: str
) -> dict[str, Any]:
    try:
        return strict_response_format(schema, operation)
    except StrictJsonSchemaError as exc:
        raise ScoringAgentError(str(exc)) from exc


def _provider_error(exc: Exception) -> tuple[str | None, str | None]:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) and isinstance(body, dict):
        # The OpenAI SDK normally unwraps the top-level `error` object, while
        # some compatible clients retain the full response body.
        error = body
    if not isinstance(error, dict):
        return None, None
    code = error.get("code")
    message = error.get("message")
    return (
        str(code)[:128] if code is not None else None,
        str(message)[:1000] if message is not None else None,
    )


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return value if isinstance(value, int) else None


def _is_retryable_model_error(exc: Exception, status_code: int | None) -> bool:
    if status_code is not None:
        return status_code in {408, 409, 425, 429} or 500 <= status_code <= 599
    # OpenAI connection/timeout errors have no HTTP response and therefore no
    # status code. Names keep this compatible with OpenAI-like SDK wrappers.
    return isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
    }


def _retry_delay_seconds(exc: Exception, settings: Settings, attempt: int) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, settings.scoring_retry_max_seconds)
    base = settings.scoring_retry_base_seconds * (2 ** (attempt - 1))
    capped = min(base, settings.scoring_retry_max_seconds)
    # Positive jitter avoids synchronized retry storms while keeping the
    # configured base delay as a guaranteed minimum.
    return min(
        capped * (1.0 + random.uniform(0.0, 0.25)),
        settings.scoring_retry_max_seconds,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, delay)


_SECTION_INSTRUCTION = """你是销售关键行为评分代理。一次处理一个销售周期内的一个聚合分析维度。
输入中的 candidate_evidence 是该销售整个日期范围的统一召回结果，不是逐通电话请求。请在一次调用中完成当前维度的全部证据判断。
只返回真正命中规则的稀疏 judgments；没有命中的通话与规则不要输出，系统会在本地补为未命中或证据不足。不得为了覆盖所有 calls 与 rules 生成大量空判断。rationale 用一句简洁中文说明。
candidate_evidence 是经过用户、销售、日期、知识命名空间过滤后，再按规则语义从向量库召回的候选证据；它仍可能不相关。
source=vector_fact 的证据来自已验证事实。source=transcript_context 表示低置信度事实只负责定位，当前内容是同一分析运行中回溯得到的原文片段；必须直接依据原文判断，不能依据低置信度事实摘要计分。
若 transcript_context 无法可靠区分销售与客户，不得据此满足“客户同意、客户反馈非负向、客户异议”等依赖说话人身份的条件，但仍可判断不依赖客户身份的明确销售话术。
只能引用 candidate_evidence 中真实存在、call_id 相同且 retrieved_for_rule_id 与 rule_id 相同的 evidence_id。不得引用相似但不能满足详细规则的内容。
一个事实只能算一次；count_capped 可引用多个彼此独立的事实，binary 最多引用一个最直接事实。
不要默认把 criteria 列表理解为必须全部同时满足。介绍业务、承上启下、挖到需求或信息、介绍卖点/优势的每条 criteria 都是可独立成立的命中路径，满足任意一项即可；只有规则解释明确要求客户反馈或处理结果时才要求组合条件。
对这四项采用贴近真实电话口语、召回优先的口径：介绍 IT/电脑/办公/硬件设备供应、租赁或相关企业服务即算介绍业务；提到之前、上次、听某人说、客户此前情况并继续当前对话即算承上启下，不强制固定出现在最初两轮；客户回答“没有需求、设备都有、不负责、找老板、项目未定”等否定或状态信息，也属于销售挖到的信息。六大卖点与实力优势都接受口语等价表达并独立计数：随时退/不用可退=随用随还，少量也能租=一台起租，零押金/不交保证金/结合先寄设备的一分钱不收=免押金或低门槛，免费上门/工程师全包/故障换机=全程保修，先用后付/按月付/一次付折扣=灵活支付，明确低月租/节省成本=高性价比，网点、规模、供货及覆盖=公司实力。不得因没有使用标准卖点名称、句子不完整或客户没有回应而漏判这四项。
涉及“客户反馈非负向”的规则必须同时有足以判断客户后续反馈的证据；没有就不要得分。
conversation_interaction 只在 computed 证据明确写出规则命中=True时引用。
audio_required 没有声学分析证据时不得凭转写文本扣分。
没有足够证据时 matched_evidence_ids=[]，并在 rationale 中说明。review_issues 非空时针对问题重新判断并返回完整替换结果。
严格按照 API 指定的 JSON Schema 返回结构化结果。"""


_REVIEW_INSTRUCTION = """你是独立评分审核代理，只审核，不直接修改分数。输入只列出整个周期内实际产生非零分数的通话与规则；零分和未召回项不需要审核。
source=vector_fact 是上游已经验证的事实，source=transcript_context 是回溯得到的原文，source=call_metric 是本地确定性计算；审核时信任这些内容确实存在，不得重新质疑事实真伪、转写真实性或召回覆盖率。
检查每个得分或扣分是否满足规则真正的必要条件、证据是否属于同一通电话、是否重复计分、客户反馈要求是否有证据、计数项是否把同一信息拆成多次，以及正负结论是否矛盾。不要把介绍业务、承上启下、挖到需求或信息、介绍卖点/优势的 criteria 误当成必须全部满足；它们是可独立命中的替代路径。接受简称、省略句、口头语、否定需求信息和语义等价卖点。
不要因为未召回到某行为、整体覆盖率低或存在其他无证据规则而质疑当前已有得分。只有引用证据在逻辑上不能满足当前规则必要条件时才报告问题。若当前结果合理，reasonable=true 且 issues=[]。
正常审核输出的 status=completed。若不合理，对每个有争议的具体判断给出 call_id、rule_id、明确缺失的规则条件，并提供0到5条只针对该通电话缺失条件的 supplemental_queries。一个 issue 只代表该 call_id 与 rule_id 的评分判断需要返工，不代表其他事实或整批结果不可信。严格按照 API 指定的 JSON Schema 返回结构化结果。"""
