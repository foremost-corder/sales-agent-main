"""Composition helpers for persistence-backed agent ports."""

from sqlalchemy.orm import Session

from sales_agent.agent.audit import RunAuditSink
from sales_agent.agent.memory import ConversationMemory
from sales_agent.repositories.agent_audit import SqlAlchemyRunAuditSink
from sales_agent.repositories.conversation_memory import SqlAlchemyConversationMemory


def build_run_audit_sink(session: Session) -> RunAuditSink:
    return SqlAlchemyRunAuditSink(session)


def build_conversation_memory(
    session: Session, *, context_messages: int
) -> ConversationMemory:
    return SqlAlchemyConversationMemory(
        session, context_messages=context_messages
    )
