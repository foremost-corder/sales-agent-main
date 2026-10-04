from __future__ import annotations

from hashlib import sha256
from typing import Any

from sales_agent.features.knowledge.contracts import (
    FactExtractionResult,
    KnowledgeDocumentSpec,
    TranscriptTurn,
)


DOCUMENT_BUILDER_VERSION = "call-documents-v2"


def build_knowledge_documents(
    *,
    call_id: str,
    raw_source_text: str,
    turns: list[TranscriptTurn],
    facts: FactExtractionResult,
    common_metadata: dict[str, Any],
    max_chunk_characters: int = 600,
    overlap_turns: int = 1,
) -> list[KnowledgeDocumentSpec]:
    documents: list[KnowledgeDocumentSpec] = []

    for ordinal, fact in enumerate(facts.facts, start=1):
        quotes = [item.quote for item in fact.evidence]
        grounding_statuses = [item.grounding for item in fact.evidence]
        turn_ids = [item.turn_no for item in fact.evidence]
        content = (
            f"评分语义标签：{'；'.join(fact.score_tags)}\n"
            f"事实：{fact.fact}\n"
            f"置信度：{fact.confidence:.2f}\n"
            f"校验状态：{fact.validation_status}\n"
            f"证据定位：{'；'.join(grounding_statuses)}\n"
            f"原文证据：{'；'.join(quotes)}"
        )
        documents.append(
            KnowledgeDocumentSpec(
                doc_type="structured_fact",
                source_key=f"fact:{fact.fact_id}",
                ordinal=ordinal,
                content=content,
                embedding_text=content,
                metadata={
                    **common_metadata,
                    "fact_id": fact.fact_id,
                    "phase": fact.phase,
                    "fact_type": fact.fact_type,
                    "speaker": fact.speaker,
                    "explicit": fact.explicit,
                    "confidence": fact.confidence,
                    "validation_status": fact.validation_status,
                    "quality_issues": fact.quality_issues,
                    "score_tags": fact.score_tags,
                    "turn_ids": turn_ids,
                    "quotes": quotes,
                    "grounding_statuses": grounding_statuses,
                },
            )
        )

    for ordinal, chunk_turns in enumerate(
        _chunk_turns(turns, max_characters=max_chunk_characters, overlap_turns=overlap_turns),
        start=1,
    ):
        content = "\n".join(_render_turn(turn) for turn in chunk_turns)
        documents.append(
            KnowledgeDocumentSpec(
                doc_type="transcript_chunk",
                source_key=f"chunk:{ordinal}",
                ordinal=ordinal,
                content=content,
                embedding_text=content,
                metadata={
                    **common_metadata,
                    "turn_ids": [turn.turn_no for turn in chunk_turns],
                    "speaker_roles": sorted({turn.speaker.value for turn in chunk_turns}),
                    "start_timestamp_ms": chunk_turns[0].timestamp_ms,
                    "end_timestamp_ms": chunk_turns[-1].timestamp_ms,
                    "chunking_version": DOCUMENT_BUILDER_VERSION,
                },
            )
        )

    full_embedding_text = "\n".join(_render_turn(turn) for turn in turns)
    documents.append(
        KnowledgeDocumentSpec(
            doc_type="full_transcript",
            source_key="full:1",
            ordinal=1,
            content=raw_source_text,
            embedding_text=full_embedding_text,
            metadata={
                **common_metadata,
                "turn_count": len(turns),
                "source_character_count": len(raw_source_text),
                "source_content_hash": sha256(raw_source_text.encode("utf-8")).hexdigest(),
            },
        )
    )
    return documents


def _chunk_turns(
    turns: list[TranscriptTurn], *, max_characters: int, overlap_turns: int
) -> list[list[TranscriptTurn]]:
    chunks: list[list[TranscriptTurn]] = []
    current: list[TranscriptTurn] = []
    current_size = 0
    for turn in turns:
        rendered_size = len(_render_turn(turn))
        if current and current_size + rendered_size + 1 > max_characters:
            chunks.append(list(current))
            current = current[-overlap_turns:] if overlap_turns else []
            current_size = sum(len(_render_turn(item)) for item in current) + max(0, len(current) - 1)
        current.append(turn)
        current_size += rendered_size + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(list(current))
    return chunks


def _render_turn(turn: TranscriptTurn) -> str:
    timestamp = f"|{turn.timestamp_ms}ms" if turn.timestamp_ms is not None else ""
    speaker = (
        turn.source_speaker_label
        if turn.speaker.value == "unknown"
        else turn.speaker.value
    )
    return f"[{turn.turn_no}|{speaker}{timestamp}] {turn.text}"
