"""SQLAlchemy conversation-memory adapter."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from sales_agent.agent.contracts import ChatMessage
from sales_agent.agent.memory import MemorySnapshot, StoredMessage
from sales_agent.domain.models import Message
from sales_agent.repositories.conversations import (
    ConversationNotFoundError,
    append_message,
    get_conversation,
    list_messages,
)


class SqlAlchemyConversationMemory:
    def __init__(self, session: Session, *, context_messages: int) -> None:
        self._session = session
        self._context_messages = context_messages

    def append(
        self,
        *,
        conversation_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
    ) -> StoredMessage:
        message = append_message(
            self._session,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        return StoredMessage(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            sequence_no=message.sequence_no,
            created_at=message.created_at,
        )

    def load(self, *, conversation_id: UUID) -> MemorySnapshot:
        conversation = get_conversation(
            self._session, conversation_id=conversation_id
        )
        if conversation is None:
            raise ConversationNotFoundError(str(conversation_id))
        history = list_messages(
            self._session,
            conversation_id=conversation_id,
            limit=self._context_messages,
            after_message_id=conversation.summary_through_message_id,
        )
        return MemorySnapshot(
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            messages=[
                ChatMessage(role=message.role, content=message.content)
                for message in history
                if message.role in {"user", "assistant"}
            ],
            summary=conversation.summary,
            summary_through_message_id=conversation.summary_through_message_id,
        )

    def save_summary(
        self,
        *,
        conversation_id: UUID,
        summary: str,
        through_message_id: UUID,
    ) -> None:
        conversation = get_conversation(
            self._session, conversation_id=conversation_id
        )
        if conversation is None:
            raise ConversationNotFoundError(str(conversation_id))
        through_message = self._session.get(Message, through_message_id)
        if (
            through_message is None
            or through_message.conversation_id != conversation_id
        ):
            raise ValueError("summary cursor does not belong to the conversation")
        conversation.summary = summary
        conversation.summary_through_message_id = through_message_id
        self._session.add(conversation)
        self._session.commit()
