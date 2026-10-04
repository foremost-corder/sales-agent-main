from uuid import uuid4
import logging

from sales_agent.features.knowledge.contracts import CallKnowledgeResult
from sales_agent.features.knowledge.standard_tool import CallKnowledgeToolExecutor
from sales_agent.features.knowledge.parser import TranscriptParseError


class FakeWorkflow:
    def __init__(self) -> None:
        self.received = None

    def execute(self, **kwargs):
        self.received = kwargs
        return CallKnowledgeResult(
            analysis_run_id=str(uuid4()),
            call_id=str(kwargs["call_id"]),
            status="completed",
            reused=False,
            fact_count=2,
            transcript_chunk_count=1,
            full_transcript_count=1,
            document_count=4,
            embedding_model="test-embedding",
        )


class FailingWorkflow:
    def execute(self, **kwargs):
        raise TranscriptParseError("bad transcript")


def test_call_knowledge_tool_is_a_thin_scoped_adapter() -> None:
    workflow = FakeWorkflow()
    executor = CallKnowledgeToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        workflow=workflow,  # type: ignore[arg-type]
    )
    call_id = uuid4()

    result = executor.execute("analyze_call", {"call_id": str(call_id)})

    assert result.ok is True
    assert result.result is not None
    assert result.result["document_count"] == 4
    assert workflow.received == {
        "call_id": call_id,
        "user_id": "user-1",
        "force": False,
        "knowledge_namespace": "baseline-react-v4",
    }
    definition = executor.tool_definitions()[0]
    assert definition.read_only is False
    assert definition.requires_approval is False


def test_call_knowledge_tool_rejects_invalid_arguments() -> None:
    result = CallKnowledgeToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        workflow=FakeWorkflow(),  # type: ignore[arg-type]
    ).execute("analyze_call", {"call_id": "not-a-uuid"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


def test_call_knowledge_tool_logs_failure_context(caplog) -> None:
    call_id = uuid4()
    executor = CallKnowledgeToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        workflow=FailingWorkflow(),  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.ERROR, logger="sales_agent.features.knowledge.standard_tool"):
        result = executor.execute("analyze_call", {"call_id": str(call_id)})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "analysis_failed"
    assert "analyze_call_tool_failed" in caplog.text
    assert str(call_id) in caplog.text
    assert "TranscriptParseError" in caplog.text
