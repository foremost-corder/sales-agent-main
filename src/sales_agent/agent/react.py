"""Graph-shaped ReAct loop with evidence-validated terminal answers."""

import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from sales_agent.agent.audit import NullRunAuditSink, RunAuditSink
from sales_agent.agent.context import build_run_context
from sales_agent.agent.contracts import AgentResult, ChatMessage, ChatModel, ToolRequest
from sales_agent.agent.evidence import (
    FINAL_ANSWER_TOOL,
    FINAL_ANSWER_TOOL_NAME,
    EvidenceLedger,
)
from sales_agent.agent.finalization import (
    direct_answer_repair_message,
    validate_finalization,
)
from sales_agent.agent.tool_runtime import ToolRuntime
from sales_agent.agent.protocol import (
    assistant_content,
    assistant_message,
    tool_message,
    tool_requests,
)


class AgentMaxStepsError(RuntimeError):
    pass


class AgentProtocolError(RuntimeError):
    pass


class ReActState(TypedDict):
    messages: list[dict[str, Any]]
    pending_calls: list[ToolRequest]
    final_answer: str | None
    terminal_error: Literal["max_steps", "final_answer_invalid"] | None
    steps: int
    tool_count: int
    final_failures: int


class ReActAgent:
    """Agent -> business tools -> evidence ledger -> validated final action."""

    def __init__(
        self,
        model: ChatModel,
        *,
        tool_runtime: ToolRuntime,
        max_steps: int,
        max_final_repairs: int = 3,
        audit_sink: RunAuditSink | None = None,
    ) -> None:
        if not 0 <= max_final_repairs <= 3:
            raise ValueError("max_final_repairs must be between 0 and 3")
        self._model = model
        self._tool_runtime = tool_runtime
        self._max_steps = max_steps
        self._max_final_repairs = max_final_repairs
        self._audit_sink = audit_sink or NullRunAuditSink()

    def run(
        self,
        *,
        conversation_id: uuid.UUID,
        messages: list[ChatMessage],
        conversation_summary: str | None = None,
    ) -> AgentResult:
        run_id = self._audit_sink.start_run(conversation_id=conversation_id)
        ledger = EvidenceLedger()

        def agent_node(state: ReActState) -> dict[str, Any]:
            response = self._model.create_completion(
                messages=state["messages"],
                tools=[*self._tool_runtime.definitions(), FINAL_ANSWER_TOOL],
            )
            normalized = assistant_message(response)
            requests = tool_requests(response)
            updated_messages = [*state["messages"], normalized]
            if requests:
                business_count = sum(
                    request.name != FINAL_ANSWER_TOOL_NAME for request in requests
                )
                if state["tool_count"] + business_count > self._max_steps:
                    return {
                        "messages": updated_messages,
                        "pending_calls": [],
                        "terminal_error": "max_steps",
                        "steps": state["steps"] + 1,
                    }
                return {
                    "messages": updated_messages,
                    "pending_calls": requests,
                    "steps": state["steps"] + 1,
                }

            failed_attempt = state["final_failures"] + 1
            content = assistant_content(response)
            if failed_attempt > self._max_final_repairs:
                return {
                    "messages": updated_messages,
                    "pending_calls": [],
                    "terminal_error": "final_answer_invalid",
                    "steps": state["steps"] + 1,
                    "final_failures": failed_attempt,
                }
            correction = direct_answer_repair_message(
                failed_attempt=failed_attempt,
                max_repairs=self._max_final_repairs,
                previous_text=content or "<empty response>",
            )
            return {
                "messages": [
                    *updated_messages,
                    {"role": "user", "content": correction},
                ],
                "pending_calls": [],
                "steps": state["steps"] + 1,
                "final_failures": failed_attempt,
            }

        def tool_node(state: ReActState) -> dict[str, Any]:
            requests = state["pending_calls"]
            tool_messages: list[dict[str, str]] = []
            final_answer: str | None = None
            terminal_error: Literal["final_answer_invalid"] | None = None
            final_failures = state["final_failures"]
            business_count = 0

            for request in requests:
                if request.name == FINAL_ANSWER_TOOL_NAME:
                    failed_attempt = final_failures + 1
                    protocol_error = None
                    if len(requests) != 1:
                        protocol_error = (
                            "submit_final_answer 必须单独调用，不能与业务工具在同一响应中调用。"
                        )
                    attempt = validate_finalization(
                        arguments=request.arguments,
                        ledger=ledger,
                        failed_attempt=failed_attempt,
                        max_repairs=self._max_final_repairs,
                        protocol_error=protocol_error,
                    )
                    self._audit_sink.record_tool_call(
                        run_id=run_id,
                        tool_name=request.name,
                        arguments=attempt.arguments,
                        result=attempt.output,
                        succeeded=bool(attempt.output["ok"]),
                    )
                    if attempt.answer is not None:
                        final_answer = attempt.answer.answer
                        continue
                    final_failures = failed_attempt
                    tool_messages.append(tool_message(request.call_id, attempt.output))
                    if final_failures > self._max_final_repairs:
                        terminal_error = "final_answer_invalid"
                    continue

                invocation = self._tool_runtime.invoke(
                    tool_name=request.name,
                    arguments=request.arguments,
                )
                evidence_ref = ledger.record(
                    call_id=request.call_id,
                    tool_name=request.name,
                    evidence_kind=invocation.evidence_kind,
                    succeeded=invocation.succeeded,
                )
                audited_output = dict(invocation.output)
                if evidence_ref is not None:
                    audited_output["evidence_ref"] = evidence_ref
                self._audit_sink.record_tool_call(
                    run_id=run_id,
                    tool_name=request.name,
                    arguments=invocation.arguments,
                    result=audited_output,
                    succeeded=invocation.succeeded,
                )
                tool_messages.append(tool_message(request.call_id, audited_output))
                business_count += 1

            return {
                "messages": [*state["messages"], *tool_messages],
                "pending_calls": [],
                "final_answer": final_answer,
                "terminal_error": terminal_error,
                "tool_count": state["tool_count"] + business_count,
                "final_failures": final_failures,
            }

        def after_agent(state: ReActState) -> Literal["tools", "agent", "end"]:
            if state["terminal_error"] or state["final_answer"] is not None:
                return "end"
            return "tools" if state["pending_calls"] else "agent"

        def after_tools(state: ReActState) -> Literal["agent", "end"]:
            return (
                "end"
                if state["terminal_error"] or state["final_answer"] is not None
                else "agent"
            )

        graph = StateGraph(ReActState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", after_agent, {"tools": "tools", "agent": "agent", "end": END}
        )
        graph.add_conditional_edges(
            "tools", after_tools, {"agent": "agent", "end": END}
        )
        compiled = graph.compile()

        initial_messages = build_run_context(messages, summary=conversation_summary)
        initial_state: ReActState = {
            "messages": initial_messages,
            "pending_calls": [],
            "final_answer": None,
            "terminal_error": None,
            "steps": 0,
            "tool_count": 0,
            "final_failures": 0,
        }
        try:
            final_state = compiled.invoke(
                initial_state,
                config={
                    "recursion_limit": (
                        self._max_steps * 2 + self._max_final_repairs * 2 + 8
                    )
                },
            )
        except Exception:
            self._audit_sink.finish_run(
                run_id=run_id,
                status="failed",
                graph_state={"steps": 0, "tool_calls": 0, "final_failures": 0},
            )
            raise

        graph_state = {
            "steps": final_state["steps"],
            "tool_calls": final_state["tool_count"],
            "final_failures": final_state["final_failures"],
        }
        answer = final_state["final_answer"]
        if answer is not None:
            self._audit_sink.finish_run(
                run_id=run_id, status="completed", graph_state=graph_state
            )
            return AgentResult(
                content=answer,
                steps=final_state["steps"],
                tool_calls=final_state["tool_count"],
            )
        if final_state["terminal_error"] == "max_steps":
            self._audit_sink.finish_run(
                run_id=run_id, status="max_steps", graph_state=graph_state
            )
            raise AgentMaxStepsError("agent exceeded maximum business tool steps")
        self._audit_sink.finish_run(
            run_id=run_id, status="failed", graph_state=graph_state
        )
        raise AgentProtocolError(
            "model failed to submit an evidence-valid final answer after bounded repairs"
        )
