"""Compilation and execution of validated semantic call queries."""

from datetime import date
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.orm import Session

from sales_agent.domain.models import Call
from sales_agent.features.calls.query import CallDimension, CallMetric, FilterOperator, QueryCallsInput, SortDirection


DIMENSION_COLUMNS: dict[CallDimension, ColumnElement[Any]] = {
    CallDimension.CALL_DATE: Call.call_date,
    CallDimension.SALES_ID: Call.sales_id,
    CallDimension.SALES_STAGE: Call.sales_stage,
    CallDimension.ANALYSIS_STATUS: Call.analysis_status,
    CallDimension.CALL_ID: Call.id,
}


@dataclass(frozen=True)
class CallContent:
    call_id: UUID
    transcript_text: str
    analysis_status: str


@dataclass(frozen=True)
class CallQueryResult:
    rows: list[dict[str, Any]]
    total_matches: int
    total_result_rows: int


def get_call_content(session: Session, *, user_id: str, call_id: UUID) -> CallContent | None:
    row = session.execute(
        select(Call.id, Call.transcript_text, Call.analysis_status).where(
            Call.id == call_id,
            Call.user_id == user_id,
        )
    ).one_or_none()
    if row is None:
        return None
    return CallContent(
        call_id=row.id,
        transcript_text=row.transcript_text,
        analysis_status=row.analysis_status,
    )


def query_calls(session: Session, *, user_id: str, query: QueryCallsInput) -> CallQueryResult:
    """Run a semantic query under a non-negotiable current-user scope."""
    metric_columns: dict[CallMetric, ColumnElement[Any]] = {
        CallMetric.CALL_COUNT: func.count(Call.id).label(CallMetric.CALL_COUNT.value),
    }
    selected_dimensions = [DIMENSION_COLUMNS[item].label(item.value) for item in query.dimensions]
    selected_metrics = [metric_columns[item] for item in query.metrics]
    statement: Select[Any] = select(*selected_dimensions, *selected_metrics).where(Call.user_id == user_id)
    match_count = select(func.count(Call.id)).where(Call.user_id == user_id)

    for condition in query.filters:
        column = DIMENSION_COLUMNS[condition.field]
        value: str | list[str] | date = condition.value
        if condition.field == CallDimension.CALL_DATE:
            assert isinstance(value, str)
            value = date.fromisoformat(value)
        if condition.operator == FilterOperator.EQ:
            statement = statement.where(column == value)
            match_count = match_count.where(column == value)
        elif condition.operator == FilterOperator.IN:
            assert isinstance(value, list)
            statement = statement.where(column.in_(value))
            match_count = match_count.where(column.in_(value))
        elif condition.operator == FilterOperator.GTE:
            statement = statement.where(column >= value)
            match_count = match_count.where(column >= value)
        elif condition.operator == FilterOperator.LTE:
            statement = statement.where(column <= value)
            match_count = match_count.where(column <= value)

    if selected_dimensions:
        statement = statement.group_by(*selected_dimensions)

    selected_by_name = {
        **{item.value: DIMENSION_COLUMNS[item] for item in query.dimensions},
        **{item.value: column for item, column in metric_columns.items()},
    }
    for ordering in query.order_by:
        column = selected_by_name[ordering.field.value]
        statement = statement.order_by(column.asc() if ordering.direction == SortDirection.ASC else column.desc())
    if not query.order_by and selected_dimensions:
        statement = statement.order_by(*selected_dimensions)

    total_matches = int(session.scalar(match_count) or 0)
    total_result_rows = int(
        session.scalar(
            select(func.count()).select_from(statement.order_by(None).subquery())
        )
        or 0
    )
    rows = session.execute(
        statement.offset(query.offset).limit(query.limit)
    ).mappings().all()
    return CallQueryResult(
        rows=[_serialize_row(dict(row)) for row in rows],
        total_matches=total_matches,
        total_result_rows=total_result_rows,
    )


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value.isoformat() if hasattr(value, "isoformat") else str(value) if key == "call_id" else value for key, value in row.items()}
