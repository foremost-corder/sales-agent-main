"""Evidence rules that are independent of a user's natural-language intent."""

from dataclasses import dataclass, field

from pydantic import ValidationError

from sales_agent.agent.contracts import FinalAnswer
from sales_agent.agent.structured_json import strict_model_json_schema
from sales_agent.agent.tool_runtime import EvidenceKind, ToolSpec


class FinalAnswerValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceRecord:
    """Server-owned binding between a short citation and transport provenance."""

    ref: str
    tool_call_id: str
    tool_name: str
    evidence_kind: EvidenceKind


@dataclass
class EvidenceLedger:
    """Assigns model-friendly evidence refs while retaining transport provenance."""

    records: dict[str, EvidenceRecord] = field(default_factory=dict)

    def record(
        self,
        *,
        call_id: str,
        tool_name: str,
        evidence_kind: EvidenceKind,
        succeeded: bool,
    ) -> str | None:
        if not succeeded or evidence_kind == "none":
            return None
        ref = f"E{len(self.records) + 1}"
        self.records[ref] = EvidenceRecord(
            ref=ref,
            tool_call_id=call_id,
            tool_name=tool_name,
            evidence_kind=evidence_kind,
        )
        return ref

    def refs_for(self, evidence_kind: EvidenceKind) -> list[str]:
        return [
            ref
            for ref, record in self.records.items()
            if record.evidence_kind == evidence_kind
        ]

    def bindings_for(self, refs: list[str]) -> list[dict[str, str]]:
        return [
            {
                "evidence_ref": record.ref,
                "tool_call_id": record.tool_call_id,
                "tool_name": record.tool_name,
                "evidence_kind": record.evidence_kind,
            }
            for ref in refs
            if (record := self.records.get(ref)) is not None
        ]

    def validate(self, payload: dict[str, object]) -> FinalAnswer:
        try:
            answer = FinalAnswer.model_validate(payload)
        except ValidationError as exc:
            issues = [
                {
                    "code": item["type"],
                    "path": list(item["loc"]),
                    "message": item["msg"],
                }
                for item in exc.errors(include_url=False, include_input=False)
            ]
            raise FinalAnswerValidationError(
                "submit_final_answer 参数不符合证据契约："
                + str(issues)
            ) from exc

        supplied = set(answer.evidence_refs)
        substantive_refs = set(self.refs_for("substantive"))
        metadata_refs = set(self.refs_for("metadata"))
        if answer.grounding == "tool":
            if not supplied or not supplied.issubset(substantive_refs):
                raise FinalAnswerValidationError("业务结论必须引用本轮成功的查询或通话文本工具调用。")
        elif answer.grounding == "metadata":
            if not supplied or not supplied.issubset(metadata_refs):
                raise FinalAnswerValidationError("查询能力说明必须引用本轮成功的字段目录工具调用。")
        elif supplied:
            raise FinalAnswerValidationError("未使用工具证据的回答不能附带证据编号。")
        return answer


FINAL_ANSWER_TOOL_NAME = "submit_final_answer"
FINAL_ANSWER_TOOL = ToolSpec(
    name=FINAL_ANSWER_TOOL_NAME,
    description=(
        "唯一允许的最终回答出口。完成必要的业务工具调用后必须单独调用本工具；"
        "业务数据结论使用 grounding=tool 并引用工具结果中的短证据编号（如 E1），"
        "字段能力说明使用 grounding=metadata，普通对话使用 grounding=none。"
    ),
    parameters=strict_model_json_schema(FinalAnswer),
)
