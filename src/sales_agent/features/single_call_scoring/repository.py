from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from sales_agent.domain.models import Call


@dataclass(frozen=True)
class SingleCallScoringSource:
    call_id: UUID
    sales_stage: str
    transcript_text: str


def get_single_call_scoring_source(
    session: Session, *, user_id: str, call_id: UUID
) -> SingleCallScoringSource | None:
    row = session.execute(
        select(Call.id, Call.sales_stage, Call.transcript_text).where(
            Call.id == call_id,
            Call.user_id == user_id,
        )
    ).one_or_none()
    if row is None:
        return None
    return SingleCallScoringSource(
        call_id=row.id,
        sales_stage=row.sales_stage,
        transcript_text=row.transcript_text,
    )
