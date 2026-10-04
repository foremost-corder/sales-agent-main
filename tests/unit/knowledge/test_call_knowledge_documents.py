from sales_agent.features.knowledge.contracts import (
    ExtractedFact,
    FactEvidence,
    FactExtractionResult,
)
from sales_agent.features.knowledge.documents import build_knowledge_documents
from sales_agent.features.knowledge.parser import parse_transcript_turns


def test_build_documents_creates_facts_turn_chunks_and_exact_full_source() -> None:
    turns = parse_transcript_turns("用户0：想了解需求\n用户1：下周需要二十台电脑")
    facts = FactExtractionResult(
        schema_version="conversation-facts-v1",
        call_id="call-1",
        facts=[
            ExtractedFact(
                fact_id="fact-1",
                phase="discovery",
                fact_type="customer_need",
                speaker="customer",
                fact="客户下周需要二十台电脑。",
                explicit=True,
                score_tags=["需求时间", "需求数量"],
                evidence=[FactEvidence(turn_no=2, quote="下周需要二十台电脑")],
            )
        ],
    )
    raw_source = "【通话ID：call-1】\n用户0：想了解需求\n用户1：下周需要二十台电脑"

    documents = build_knowledge_documents(
        call_id="call-1",
        raw_source_text=raw_source,
        turns=turns,
        facts=facts,
        common_metadata={"user_id": "user-1"},
    )

    assert [item.doc_type for item in documents] == [
        "structured_fact",
        "transcript_chunk",
        "full_transcript",
    ]
    assert documents[0].metadata["turn_ids"] == [2]
    assert documents[0].metadata["confidence"] == 1.0
    assert documents[0].metadata["validation_status"] == "verified"
    assert documents[1].metadata["turn_ids"] == [1, 2]
    assert documents[2].content == raw_source
    assert documents[2].metadata["turn_count"] == 2


def test_unresolved_turns_keep_source_labels_in_raw_documents() -> None:
    turns = parse_transcript_turns("用户0：你好\n用户0：暂时不需要")
    documents = build_knowledge_documents(
        call_id="call-2",
        raw_source_text="用户0：你好\n用户0：暂时不需要",
        turns=turns,
        facts=FactExtractionResult(
            schema_version="conversation-facts-v1",
            call_id="call-2",
            facts=[],
            status="skipped",
        ),
        common_metadata={"degraded": True},
    )

    chunk = next(item for item in documents if item.doc_type == "transcript_chunk")
    assert "[1|用户0] 你好" in chunk.content
    assert "[1|unknown]" not in chunk.content
