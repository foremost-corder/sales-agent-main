import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(ApiModel):
    user_id: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=500)


class ConversationResponse(ApiModel):
    id: uuid.UUID
    user_id: str
    title: str | None
    summary: str | None
    created_at: datetime
    updated_at: datetime


class MessageCreate(ApiModel):
    content: str = Field(min_length=1, max_length=20000)


class MessageResponse(ApiModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    sequence_no: int
    created_at: datetime


class MessageExchangeResponse(ApiModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
