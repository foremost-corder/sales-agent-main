from collections.abc import Iterator
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from sales_agent.api.app import app
from sales_agent.agent.chat_model import get_chat_model
from sales_agent.agent.evidence import FINAL_ANSWER_TOOL_NAME
from sales_agent.agent.tool_runtime import ToolSpec
from sales_agent.domain.models import Conversation, Message
from sales_agent.core.database import SessionLocal


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def create_completion(
        self, *, messages: list[dict[str, object]], tools: list[ToolSpec]
    ) -> SimpleNamespace:
        self.calls.append(messages)
        final_tool = next(
            tool for tool in tools if tool.name == FINAL_ANSWER_TOOL_NAME
        )
        assert final_tool.parameters["required"] == [
            "schema_version",
            "answer",
            "grounding",
            "evidence_refs",
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_final",
                                function=SimpleNamespace(
                                    name=FINAL_ANSWER_TOOL_NAME,
                                    arguments=(
                                        '{"schema_version":"agent-final-answer-v1",'
                                        '"answer":"这是测试模型回复",'
                                        '"grounding":"none","evidence_refs":[]}'
                                    ),
                                ),
                            )
                        ],
                    )
                )
            ]
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    fake_model = FakeChatModel()
    app.dependency_overrides[get_chat_model] = lambda: fake_model
    app.state.fake_chat_model = fake_model
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        del app.state.fake_chat_model


@pytest.mark.integration
def test_create_conversation(client: TestClient) -> None:
    user_id = f"test-user-{uuid4()}"

    response = client.post(
        "/api/conversations",
        json={"user_id": user_id, "title": "First test conversation"},
    )

    try:
        assert response.status_code == 201
        body = response.json()
        assert body["user_id"] == user_id
        assert body["title"] == "First test conversation"
        assert body["summary"] is None
        assert body["id"]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_list_conversations_for_user(client: TestClient) -> None:
    user_id = f"test-user-{uuid4()}"
    other_user_id = f"test-user-{uuid4()}"
    try:
        expected = client.post(
            "/api/conversations",
            json={"user_id": user_id, "title": "Visible conversation"},
        ).json()
        client.post(
            "/api/conversations",
            json={"user_id": other_user_id, "title": "Hidden conversation"},
        )

        response = client.get(
            "/api/conversations", params={"user_id": user_id, "limit": 50}
        )

        assert response.status_code == 200
        assert [item["id"] for item in response.json()] == [expected["id"]]
    finally:
        with SessionLocal.begin() as session:
            session.execute(
                delete(Conversation).where(
                    Conversation.user_id.in_([user_id, other_user_id])
                )
            )


@pytest.mark.integration
def test_send_message_persists_user_and_assistant_messages(client: TestClient) -> None:
    user_id = f"test-user-{uuid4()}"
    create_response = client.post(
        "/api/conversations",
        json={"user_id": user_id, "title": "Persistence test"},
    )
    conversation_id = create_response.json()["id"]

    try:
        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "请记住这条测试消息"},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["user_message"]["role"] == "user"
        assert body["user_message"]["content"] == "请记住这条测试消息"
        assert body["user_message"]["sequence_no"] == 1
        assert body["assistant_message"]["role"] == "assistant"
        assert body["assistant_message"]["sequence_no"] == 2

        with SessionLocal() as session:
            messages = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_no)
            ).all()
            assert [(message.role, message.sequence_no) for message in messages] == [
                ("user", 1),
                ("assistant", 2),
            ]
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_send_message_to_missing_conversation_returns_404(client: TestClient) -> None:
    response = client.post(
        f"/api/conversations/{uuid4()}/messages",
        json={"content": "This conversation does not exist"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation not found"}


@pytest.mark.integration
def test_read_conversation_and_ordered_message_history(client: TestClient) -> None:
    user_id = f"test-user-{uuid4()}"
    create_response = client.post(
        "/api/conversations",
        json={"user_id": user_id, "title": "History test"},
    )
    conversation_id = create_response.json()["id"]

    try:
        for content in ("第一轮消息", "第二轮消息"):
            response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"content": content},
            )
            assert response.status_code == 201

        detail_response = client.get(f"/api/conversations/{conversation_id}")
        assert detail_response.status_code == 200
        assert detail_response.json()["title"] == "History test"

        history_response = client.get(
            f"/api/conversations/{conversation_id}/messages"
        )
        assert history_response.status_code == 200
        history = history_response.json()
        assert [message["sequence_no"] for message in history] == [1, 2, 3, 4]
        assert [message["role"] for message in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert history[0]["content"] == "第一轮消息"
        assert history[2]["content"] == "第二轮消息"
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))


@pytest.mark.integration
def test_model_context_is_limited_to_recent_twelve_messages(client: TestClient) -> None:
    user_id = f"test-user-{uuid4()}"
    conversation_id = client.post(
        "/api/conversations",
        json={"user_id": user_id, "title": "Context limit test"},
    ).json()["id"]

    try:
        for number in range(1, 8):
            response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"content": f"message-{number}"},
            )
            assert response.status_code == 201

        latest_context = client.app.state.fake_chat_model.calls[-1]
        assert len(latest_context) == 12
        assert latest_context[-1] == {"role": "user", "content": "message-7"}
        assert {"role": "user", "content": "message-1"} not in latest_context
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Conversation).where(Conversation.user_id == user_id))
