import pytest

from sales_agent.agent.evidence import EvidenceLedger, FinalAnswerValidationError


def test_metadata_cannot_be_claimed_as_business_data_evidence() -> None:
    ledger = EvidenceLedger()
    ledger.record(call_id="call_schema", tool_name="describe_calls_schema", evidence_kind="metadata", succeeded=True)

    with pytest.raises(FinalAnswerValidationError, match="业务结论"):
        ledger.validate(
            {
                "schema_version": "agent-final-answer-v1",
                "answer": "销售001有通话",
                "grounding": "tool",
                "evidence_refs": ["E1"],
            }
        )


def test_query_result_can_ground_a_business_answer() -> None:
    ledger = EvidenceLedger()
    ref = ledger.record(call_id="call_query", tool_name="query_calls", evidence_kind="substantive", succeeded=True)

    answer = ledger.validate(
        {
            "schema_version": "agent-final-answer-v1",
            "answer": "销售001在2026-09-03有通话。",
            "grounding": "tool",
            "evidence_refs": ["E1"],
        }
    )

    assert answer.answer.startswith("销售001")
    assert ref == "E1"
    assert ledger.bindings_for(answer.evidence_refs) == [
        {
            "evidence_ref": "E1",
            "tool_call_id": "call_query",
            "tool_name": "query_calls",
            "evidence_kind": "substantive",
        }
    ]


def test_knowledge_read_results_can_ground_a_business_answer() -> None:
    ledger = EvidenceLedger()
    ledger.record(call_id="call_facts", tool_name="get_call_facts", evidence_kind="substantive", succeeded=True)
    ledger.record(call_id="call_search", tool_name="search_call_knowledge", evidence_kind="substantive", succeeded=True)

    answer = ledger.validate(
        {
            "schema_version": "agent-final-answer-v1",
            "answer": "客户需要二十台电脑。",
            "grounding": "tool",
            "evidence_refs": ["E1", "E2"],
        }
    )

    assert answer.grounding == "tool"


def test_schema_lookup_can_ground_a_metadata_answer() -> None:
    ledger = EvidenceLedger()
    ledger.record(call_id="call_schema", tool_name="describe_calls_schema", evidence_kind="metadata", succeeded=True)

    answer = ledger.validate(
        {
            "schema_version": "agent-final-answer-v1",
            "answer": "可以查询通话日期和销售员。",
            "grounding": "metadata",
            "evidence_refs": ["E1"],
        }
    )

    assert answer.grounding == "metadata"


def test_invalid_final_schema_reports_repairable_field_errors() -> None:
    ledger = EvidenceLedger()

    with pytest.raises(FinalAnswerValidationError) as captured:
        ledger.validate({"answer": "缺少证据字段"})

    message = str(captured.value)
    assert "grounding" in message
    assert "evidence_refs" in message
