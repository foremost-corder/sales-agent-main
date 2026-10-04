"""Validated terminal action and bounded repair feedback for final answers."""

from dataclasses import dataclass

from sales_agent.agent.contracts import FinalAnswer
from sales_agent.agent.evidence import EvidenceLedger, FinalAnswerValidationError


@dataclass(frozen=True)
class FinalizationAttempt:
    arguments: dict[str, object]
    answer: FinalAnswer | None
    output: dict[str, object]


def validate_finalization(
    *,
    arguments: dict[str, object],
    ledger: EvidenceLedger,
    failed_attempt: int,
    max_repairs: int,
    protocol_error: str | None = None,
) -> FinalizationAttempt:
    try:
        if protocol_error:
            raise FinalAnswerValidationError(protocol_error)
        answer = ledger.validate(arguments)
    except FinalAnswerValidationError as exc:
        output: dict[str, object] = {
            "tool_name": "submit_final_answer",
            "ok": False,
            "result": {
                "repair_context": {
                    "failed_attempt": failed_attempt,
                    "repairs_remaining": max(0, max_repairs - failed_attempt + 1),
                    "valid_substantive_evidence_refs": ledger.refs_for("substantive"),
                    "valid_metadata_evidence_refs": ledger.refs_for("metadata"),
                }
            },
            "error": {
                "code": "final_answer_validation_failed",
                "message": str(exc),
            },
        }
        return FinalizationAttempt(arguments=arguments, answer=None, output=output)

    return FinalizationAttempt(
        arguments=arguments,
        answer=answer,
        output={
            "tool_name": "submit_final_answer",
            "ok": True,
            "result": {
                "accepted": True,
                "evidence_bindings": ledger.bindings_for(answer.evidence_refs),
            },
            "error": None,
        },
    )


def direct_answer_repair_message(
    *, failed_attempt: int, max_repairs: int, previous_text: str
) -> str:
    return (
        "最终回答协议校验失败：不得直接输出普通文本，必须调用 "
        "submit_final_answer。请提交完整的 answer、grounding 和 "
        "evidence_refs，并使用 schema_version=agent-final-answer-v1；"
        "引用工具证据时只能使用工具结果中返回的 E1、E2 等编号。"
        f"这是第 {failed_attempt} 次失败，最多允许 {max_repairs} 次返工。"
        f"上一输出仅供修复，不得直接复述：{previous_text[:500]}"
    )
