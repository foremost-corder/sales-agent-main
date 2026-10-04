from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import delete

from sales_agent.domain.models import Call
from sales_agent.core.database import SessionLocal
from sales_agent.features.calls.tool import CallSemanticToolExecutor


def make_call(*, user_id: str, call_date: date, sales_id: str) -> Call:
    return Call(
        user_id=user_id, external_call_id=str(uuid4()), sales_id=sales_id, call_date=call_date,
        sales_stage="销售线索", source_filename="test.txt", source_encoding="utf-8",
        raw_source_text="不应被自由查询的原始文本", transcript_text="不应被自由查询的通话文本",
        source_hash="a" * 64, analysis_status="pending",
    )


@pytest.mark.integration
def test_semantic_query_groups_filters_and_scopes_to_authenticated_user() -> None:
    user_id = f"semantic-user-{uuid4()}"
    other_user_id = f"semantic-user-{uuid4()}"
    target_date = date(2026, 9, 3)
    try:
        with SessionLocal.begin() as session:
            session.add_all([
                make_call(user_id=user_id, call_date=target_date, sales_id="001"),
                make_call(user_id=user_id, call_date=target_date, sales_id="001"),
                make_call(user_id=user_id, call_date=target_date, sales_id="002"),
                make_call(user_id=other_user_id, call_date=target_date, sales_id="001"),
            ])
        with SessionLocal() as session:
            result = CallSemanticToolExecutor(session, user_id=user_id).execute(
                "query_calls",
                {
                    "metrics": ["call_count"], "dimensions": ["sales_id"],
                    "filters": [{"field": "call_date", "operator": "eq", "value": "2026-09-03"}],
                    "order_by": [{"field": "call_count", "direction": "desc"}], "limit": 100,
                },
            )
        assert result.ok is True
        assert result.result is not None
        assert result.result["rows"] == [{"sales_id": "001", "call_count": 2}, {"sales_id": "002", "call_count": 1}]
        assert result.result["total_matches"] == 3
        assert result.result["total_result_rows"] == 2
        assert result.result["returned_count"] == 2
        assert result.result["has_more"] is False
        assert result.result["next_offset"] is None
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id.in_([user_id, other_user_id])))


def test_semantic_catalog_rejects_sensitive_fields_and_unsupported_sql() -> None:
    executor = CallSemanticToolExecutor(session=None, user_id="local-demo-user")  # type: ignore[arg-type]
    catalog = executor.execute("describe_calls_schema", {})
    assert catalog.ok is True
    assert catalog.result is not None
    assert "transcript_text" in catalog.result["excluded_fields"]
    sensitive = executor.execute("query_calls", {"metrics": ["call_count"], "dimensions": ["transcript_text"]})
    assert sensitive.ok is False
    assert sensitive.error is not None
    assert sensitive.error.code == "invalid_arguments"
    sql = executor.execute("execute_sql", {"sql": "SELECT * FROM calls"})
    assert sql.ok is False
    assert sql.error is not None
    assert sql.error.code == "unknown_tool"


@pytest.mark.integration
def test_get_call_content_is_scoped_and_chunked() -> None:
    user_id = f"content-user-{uuid4()}"
    other_user_id = f"content-user-{uuid4()}"
    transcript = "甲" * 700 + "乙" * 700
    try:
        with SessionLocal.begin() as session:
            call = make_call(user_id=user_id, call_date=date(2026, 9, 3), sales_id="001")
            call.transcript_text = transcript
            other = make_call(user_id=other_user_id, call_date=date(2026, 9, 3), sales_id="001")
            session.add_all([call, other])

        with SessionLocal() as session:
            tools = CallSemanticToolExecutor(session, user_id=user_id)
            first = tools.execute(
                "get_call_content",
                {"call_id": str(call.id), "start_char": 0, "max_chars": 500},
            )
            hidden = tools.execute("get_call_content", {"call_id": str(other.id)})

        assert first.ok is True
        assert first.result is not None
        assert first.result["content"] == "甲" * 500
        assert first.result["next_cursor"] == 500
        assert hidden.ok is False
        assert hidden.error is not None
        assert hidden.error.code == "not_found"
    finally:
        with SessionLocal.begin() as session:
            session.execute(delete(Call).where(Call.user_id.in_([user_id, other_user_id])))
