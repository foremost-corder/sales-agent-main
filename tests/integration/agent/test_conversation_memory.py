from uuid import uuid4

import pytest
from sqlalchemy import delete

from sales_agent.core.database import SessionLocal
from sales_agent.domain.models import Conversation
from sales_agent.repositories.conversation_memory import SqlAlchemyConversationMemory
from sales_agent.repositories.conversations import create_conversation


@pytest.mark.integration
def test_memory_load_excludes_messages_already_covered_by_summary() -> None:
    user_id = f"memory-user-{uuid4()}"
    try:
        with SessionLocal() as session:
            conversation = create_conversation(
                session, user_id=user_id, title="Memory test"
            )
            memory = SqlAlchemyConversationMemory(session, context_messages=10)
            first = memory.append(
                conversation_id=conversation.id,
                role="user",
                content="已经被总结的问题",
            )
            memory.append(
                conversation_id=conversation.id,
                role="assistant",
                content="已经被总结的回答",
            )
            memory.save_summary(
                conversation_id=conversation.id,
                summary="用户此前询问了一个已经回答的问题。",
                through_message_id=first.id,
            )

            snapshot = memory.load(conversation_id=conversation.id)

        assert snapshot.summary == "用户此前询问了一个已经回答的问题。"
        assert [message.content for message in snapshot.messages] == [
            "已经被总结的回答"
        ]
        assert snapshot.summary_through_message_id == first.id
    finally:
        with SessionLocal.begin() as session:
            session.execute(
                delete(Conversation).where(Conversation.user_id == user_id)
            )
