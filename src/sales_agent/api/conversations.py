import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from sales_agent.agent.chat_model import (
    ChatModel,
    ChatModelContractError,
    ChatModelNotConfiguredError,
    ChatModelRequestError,
    get_chat_model,
)
from sales_agent.agent.react import AgentMaxStepsError, AgentProtocolError, ReActAgent
from sales_agent.api.schemas import (
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageExchangeResponse,
    MessageResponse,
)
from sales_agent.bootstrap.agent_runtime import (
    build_conversation_memory,
    build_run_audit_sink,
)
from sales_agent.bootstrap.tools import build_tool_runtime, get_external_tool_providers
from sales_agent.core.database import get_db_session
from sales_agent.core.config import get_settings
from sales_agent.repositories.conversations import (
    ConversationNotFoundError,
    create_conversation,
    get_conversation,
    list_conversations,
    list_messages,
)
from sales_agent.tools.contracts import ToolProvider

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
DatabaseSession = Annotated[Session, Depends(get_db_session)]
ChatModelDependency = Annotated[ChatModel, Depends(get_chat_model)]
ToolProvidersDependency = Annotated[
    tuple[ToolProvider, ...], Depends(get_external_tool_providers)
]


@router.get("", response_model=list[ConversationResponse])
def get_conversations(
    session: DatabaseSession,
    user_id: Annotated[str, Query(min_length=1, max_length=255)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ConversationResponse]:
    conversations = list_conversations(session, user_id=user_id, limit=limit)
    return [
        ConversationResponse.model_validate(conversation)
        for conversation in conversations
    ]


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def post_conversation(
    payload: ConversationCreate,
    session: DatabaseSession,
) -> ConversationResponse:
    conversation = create_conversation(
        session,
        user_id=payload.user_id,
        title=payload.title,
    )
    return ConversationResponse.model_validate(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation_detail(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
) -> ConversationResponse:
    conversation = get_conversation(session, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="conversation not found",
        )
    return ConversationResponse.model_validate(conversation)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def get_message_history(
    conversation_id: uuid.UUID,
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[MessageResponse]:
    conversation = get_conversation(session, conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="conversation not found",
        )
    messages = list_messages(
        session,
        conversation_id=conversation_id,
        limit=limit,
    )
    return [MessageResponse.model_validate(message) for message in messages]


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageExchangeResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_message(
    conversation_id: uuid.UUID,
    payload: MessageCreate,
    session: DatabaseSession,
    chat_model: ChatModelDependency,
    tool_providers: ToolProvidersDependency,
) -> MessageExchangeResponse:
    settings = get_settings()
    memory = build_conversation_memory(
        session, context_messages=settings.conversation_context_messages
    )
    try:
        user_message = memory.append(
            conversation_id=conversation_id,
            role="user",
            content=payload.content,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="conversation not found",
        ) from exc

    snapshot = memory.load(conversation_id=conversation_id)
    agent = ReActAgent(
        chat_model,
        tool_runtime=build_tool_runtime(
            session,
            user_id=snapshot.user_id,
            external_providers=tool_providers,
        ),
        max_steps=settings.agent_max_steps,
        max_final_repairs=settings.agent_final_answer_max_repairs,
        audit_sink=build_run_audit_sink(session),
    )
    try:
        result = agent.run(
            conversation_id=conversation_id,
            messages=snapshot.messages,
            conversation_summary=snapshot.summary,
        )
    except ChatModelNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型尚未配置，请在 .env 中设置 OPENAI_API_KEY",
        ) from exc
    except ChatModelRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.public_detail,
        ) from exc
    except ChatModelContractError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型工具配置错误，请联系项目维护者检查工具定义",
        ) from exc
    except AgentMaxStepsError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="模型调用工具次数超过限制，请缩小问题范围后重试",
        ) from exc
    except AgentProtocolError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="模型未遵守工具调用协议，请重试",
        ) from exc

    assistant_message = memory.append(
        conversation_id=conversation_id,
        role="assistant",
        content=result.content,
    )

    return MessageExchangeResponse(
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message),
    )
