from types import SimpleNamespace
from uuid import uuid4

from sales_agent.agent.contracts import ChatMessage
from sales_agent.agent.react import ReActAgent
from sales_agent.agent.tool_runtime import ToolInvocation, ToolSpec


class FinalOnlyModel:
    def create_completion(self, *, messages, tools):
        assert all(isinstance(tool, ToolSpec) for tool in tools)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="final-1",
                                type="function",
                                function=SimpleNamespace(
                                    name="submit_final_answer",
                                    arguments=(
                                        '{"schema_version":"agent-final-answer-v1",'
                                        '"answer":"你好","grounding":"none",'
                                        '"evidence_refs":[]}'
                                    ),
                                ),
                            )
                        ],
                    )
                )
            ]
        )


class EmptyToolRuntime:
    def definitions(self) -> list[ToolSpec]:
        return []

    def invoke(self, *, tool_name: str, arguments: dict) -> ToolInvocation:
        raise AssertionError("no business tool should be invoked")


def test_react_core_runs_without_database_or_persistence_adapter() -> None:
    result = ReActAgent(
        FinalOnlyModel(),
        tool_runtime=EmptyToolRuntime(),
        max_steps=1,
    ).run(
        conversation_id=uuid4(),
        messages=[ChatMessage(role="user", content="你好")],
    )

    assert result.content == "你好"
    assert result.tool_calls == 0
