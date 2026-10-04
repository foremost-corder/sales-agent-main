"""Conversation-memory contracts independent of database persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from sales_agent.agent.contracts import ChatMessage


@dataclass(frozen=True)
class StoredMessage:
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    sequence_no: int
    created_at: datetime


@dataclass(frozen=True)
class MemorySnapshot:
    conversation_id: UUID
    user_id: str
    messages: list[ChatMessage]
    summary: str | None
    summary_through_message_id: UUID | None


class ConversationMemory(Protocol):
    def append(
        self,
        *,
        conversation_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
    ) -> StoredMessage: ...

    def load(self, *, conversation_id: UUID) -> MemorySnapshot: ...

    def save_summary(
        self,
        *,
        conversation_id: UUID,
        summary: str,
        through_message_id: UUID,
    ) -> None: ...
