import pytest
from sqlalchemy import inspect, text

from sales_agent.core.database import engine


@pytest.mark.integration
def test_database_connection_and_core_schema() -> None:
    with engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1

    table_names = set(inspect(engine).get_table_names())
    assert {
        "alembic_version",
        "conversations",
        "messages",
        "agent_runs",
        "tool_calls",
        "calls",
        "call_analysis_runs",
        "call_turns",
        "call_facts",
        "documents",
        "document_embeddings",
    }.issubset(table_names)
