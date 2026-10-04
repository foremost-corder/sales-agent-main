from uuid import uuid4

from sales_agent.features.knowledge.contracts import CallKnowledgeResult
from sales_agent.features.knowledge.lean_tool import LeanCallKnowledgeToolExecutor


class FakeLeanWorkflow:
    def __init__(self) -> None:
        self.received = None

    def execute(self, **kwargs):
        self.received = kwargs
        return CallKnowledgeResult(
            analysis_run_id=str(uuid4()),
            call_id=str(kwargs["call_id"]),
            status="completed",
            reused=False,
            fact_count=1,
            transcript_chunk_count=1,
            full_transcript_count=1,
            document_count=3,
            embedding_model="test-embedding",
        )


def test_lean_tool_has_a_distinct_name_and_delegates_to_lean_workflow() -> None:
    workflow = FakeLeanWorkflow()
    executor = LeanCallKnowledgeToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        workflow=workflow,  # type: ignore[arg-type]
    )
    call_id = uuid4()

    result = executor.execute("analyze_call_lean", {"call_id": str(call_id)})

    assert result.ok is True
    assert workflow.received == {
        "call_id": call_id,
        "user_id": "user-1",
        "force": False,
        "knowledge_namespace": "experiment-lean-v1",
    }
    assert executor.tool_definitions()[0].name == "analyze_call_lean"


def test_lean_tool_does_not_accept_the_standard_tool_name() -> None:
    result = LeanCallKnowledgeToolExecutor(
        session=None,  # type: ignore[arg-type]
        user_id="user-1",
        workflow=FakeLeanWorkflow(),  # type: ignore[arg-type]
    ).execute("analyze_call", {"call_id": str(uuid4())})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "unknown_tool"
