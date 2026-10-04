import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from sales_agent.agent.contracts import ChatMessage
from sales_agent.agent.tool_runtime import ToolSpec
from sales_agent.agent.react import AgentProtocolError, ReActAgent
from sales_agent.bootstrap.tools import build_tool_runtime
from sales_agent.domain.models import AgentRun, Conversation, ToolCall
from sales_agent.core.database import SessionLocal
from sales_agent.repositories.conversations import create_conversation
from sales_agent.repositories.agent_audit import SqlAlchemyRunAuditSink


class ScriptedModel:
    def __init__(self) -> None:
        self.requests: list[list[object]] = []

    def create_completion(
        self, *, messages: list[dict[str, object]], tools: list[ToolSpec]
    ) -> SimpleNamespace:
        self.requests.append(messages)
        if len(self.requests) == 1:
            assert {tool.name for tool in tools} == {
                "describe_calls_schema",
                "query_calls",
                "get_call_content",
                "analyze_call",
                "analyze_call_lean",
                "get_call_facts",
                "search_call_knowledge",
                "score_single_call_whiteboard",
                "submit_final_answer",
            }
            return _tool_response("call_schema", "describe_calls_schema", "{}")
        assert messages[0] == {"role": "user", "content": "有哪些通话字段？"}
        if len(self.requests) == 2:
            output = messages[-1]
            assert output["role"] == "tool"
            assert json.loads(output["content"])["ok"] is True
            return _tool_response("call_query", "query_calls", '{"metrics":["call_count"],"dimensions":["call_date"],"filters":[],"order_by":[],"limit":10}')
        tool_outputs = [item for item in messages if item.get("role") == "tool"]
        assert len(tool_outputs) == 2
        return _tool_response(
            "call_final",
            "submit_final_answer",
            json.dumps(
                {
                    "schema_version": "agent-final-answer-v1",
                    "answer": "没有匹配的通话记录。",
                    "grounding": "tool",
                    "evidence_refs": ["E2"],
                },
                ensure_ascii=False,
            ),
        )


def _tool_response(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))]))]
    )


class DirectAnswerModel:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def create_completion(
        self, *, messages: list[dict[str, object]], tools: list[ToolSpec]
    ) -> SimpleNamespace:
        self.calls.append(messages)
        return _tool_response(
            "call_final",
            "submit_final_answer",
            '{"schema_version":"agent-final-answer-v1","answer":"你好！","grounding":"none","evidence_refs":[]}',
        )


class InvalidCitationThenRepairModel:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def create_completion(self, *, messages, tools):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return _tool_response(
                "bad_final",
                "submit_final_answer",
                '{"schema_version":"agent-final-answer-v1","answer":"查到了。","grounding":"tool",'
                '"evidence_refs":["E99"]}',
            )
        error_message = messages[-1]
        assert error_message["role"] == "tool"
        error = json.loads(error_message["content"])
        assert error["error"]["code"] == "final_answer_validation_failed"
        assert error["result"]["repair_context"]["valid_substantive_evidence_refs"] == []
        return _tool_response(
            "fixed_final",
            "submit_final_answer",
            '{"schema_version":"agent-final-answer-v1","answer":"目前没有可引用的查询结果。","grounding":"none",'
            '"evidence_refs":[]}',
        )


class AlwaysInvalidFinalModel:
    def __init__(self) -> None:
        self.call_count = 0

    def create_completion(self, *, messages, tools):
        self.call_count += 1
        return _tool_response(
            f"bad_final_{self.call_count}",
            "submit_final_answer",
            '{"schema_version":"agent-final-answer-v1","answer":"无依据结论","grounding":"tool",'
            '"evidence_refs":["E99"]}',
        )


@pytest.mark.integration
def test_react_agent_executes_and_audits_a_tool_call() -> None:
    user_id = f"react-user-{uuid4()}"
    try:
        with SessionLocal() as session:
            conversation = create_conversation(session, user_id=user_id, title="ReAct test")
            result = ReActAgent(
                ScriptedModel(),
                tool_runtime=build_tool_runtime(session, user_id=user_id),
                max_steps=3,
                audit_sink=SqlAlchemyRunAuditSink(session),
            ).run(
                conversation_id=conversation.id,
                messages=[ChatMessage(role="user", content="有哪些通话字段？")],
            )
            runs = session.scalars(
                select(AgentRun).where(AgentRun.conversation_id == conversation.id)
            ).all()
            calls = session.scalars(
                select(ToolCall).where(ToolCall.run_id == runs[0].id)
            ).all()

        assert result.content == "没有匹配的通话记录。"
        assert result.steps == 3
        assert result.tool_calls == 2
        assert runs[0].status == "completed"
        assert [call.tool_name for call in calls] == [
            "describe_calls_schema",
            "query_calls",
            "submit_final_answer",
        ]
        assert all(call.status == "completed" for call in calls)
        assert calls[0].result_json["evidence_ref"] == "E1"
        assert calls[1].result_json["evidence_ref"] == "E2"
        bindings = calls[2].result_json["result"]["evidence_bindings"]
        assert bindings == [
            {
                "evidence_ref": "E2",
                "tool_call_id": "call_query",
                "tool_name": "query_calls",
                "evidence_kind": "substantive",
            }
        ]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_react_agent_can_answer_directly_without_a_tool() -> None:
    user_id = f"react-repair-{uuid4()}"
    try:
        with SessionLocal() as session:
            conversation = create_conversation(session, user_id=user_id, title="ReAct repair test")
            result = ReActAgent(
                DirectAnswerModel(),
                tool_runtime=build_tool_runtime(session, user_id=user_id),
                max_steps=3,
                audit_sink=SqlAlchemyRunAuditSink(session),
            ).run(
                conversation_id=conversation.id,
                messages=[ChatMessage(role="user", content="你好")],
            )

        assert result.content == "你好！"
        assert result.steps == 1
        assert result.tool_calls == 0
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_invalid_final_citation_is_injected_and_repaired() -> None:
    user_id = f"react-final-repair-{uuid4()}"
    try:
        with SessionLocal() as session:
            conversation = create_conversation(session, user_id=user_id, title="Final repair")
            model = InvalidCitationThenRepairModel()
            result = ReActAgent(
                model,
                tool_runtime=build_tool_runtime(session, user_id=user_id),
                max_steps=3,
                max_final_repairs=3,
                audit_sink=SqlAlchemyRunAuditSink(session),
            ).run(
                conversation_id=conversation.id,
                messages=[ChatMessage(role="user", content="给我一个结论")],
            )
            run = session.scalar(
                select(AgentRun).where(AgentRun.conversation_id == conversation.id)
            )
            calls = session.scalars(
                select(ToolCall).where(ToolCall.run_id == run.id).order_by(ToolCall.created_at)
            ).all()

        assert result.content == "目前没有可引用的查询结果。"
        assert result.steps == 2
        assert run.graph_state["final_failures"] == 1
        assert [call.status for call in calls] == ["failed", "completed"]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_invalid_final_answer_stops_after_bounded_repairs() -> None:
    user_id = f"react-final-exhausted-{uuid4()}"
    try:
        with SessionLocal() as session:
            conversation = create_conversation(session, user_id=user_id, title="Final exhausted")
            model = AlwaysInvalidFinalModel()
            with pytest.raises(AgentProtocolError, match="evidence-valid"):
                ReActAgent(
                    model,
                    tool_runtime=build_tool_runtime(session, user_id=user_id),
                    max_steps=3,
                    max_final_repairs=2,
                    audit_sink=SqlAlchemyRunAuditSink(session),
                ).run(
                    conversation_id=conversation.id,
                    messages=[ChatMessage(role="user", content="编造一个结论")],
                )
            run = session.scalar(
                select(AgentRun).where(AgentRun.conversation_id == conversation.id)
            )

        assert model.call_count == 3
        assert run.status == "failed"
        assert run.graph_state["final_failures"] == 3
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))
