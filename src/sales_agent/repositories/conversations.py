import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sales_agent.domain.models import Conversation, Message


class ConversationNotFoundError(LookupError):
    pass


def create_conversation(
    session: Session,
    *,
    user_id: str,
    title: str | None,
) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


def list_conversations(
    session: Session,
    *,
    user_id: str,
    limit: int,
) -> list[Conversation]:
    return list(
        session.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            .limit(limit)
        ).all()
    )


def get_conversation(
    session: Session,
    *,
    conversation_id: uuid.UUID,
) -> Conversation | None:
    return session.get(Conversation, conversation_id)


def list_messages(
    session: Session,
    *,
    conversation_id: uuid.UUID,
    limit: int,
    after_message_id: uuid.UUID | None = None,
) -> list[Message]:
    query = select(Message).where(Message.conversation_id == conversation_id)
    if after_message_id is not None:
        through_sequence = session.scalar(
            select(Message.sequence_no).where(
                Message.id == after_message_id,
                Message.conversation_id == conversation_id,
            )
        )
        if through_sequence is None:
            raise ValueError("summary cursor does not belong to the conversation")
        query = query.where(Message.sequence_no > through_sequence)
    recent_messages = session.scalars(
        query
        .order_by(Message.sequence_no.desc())
        .limit(limit)
    ).all()
    return list(reversed(recent_messages))


def append_message(
    session: Session,
    *,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> Message:
    conversation = session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .with_for_update()
    )
    if conversation is None:
        raise ConversationNotFoundError(str(conversation_id))

    next_sequence = session.scalar(
        select(func.coalesce(func.max(Message.sequence_no), 0) + 1).where(
            Message.conversation_id == conversation_id
        )
    )
    assert next_sequence is not None

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sequence_no=next_sequence,
    )
    if role == "user" and (conversation.title is None or conversation.title == "新对话"):
        conversation.title = content[:40]
    conversation.updated_at = func.now()
    session.add(message)
    session.commit()
    session.refresh(message)
    return message
