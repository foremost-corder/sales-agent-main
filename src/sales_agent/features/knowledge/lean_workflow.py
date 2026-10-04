"""Independent low-call variant of the call knowledge ingestion workflow."""

from sqlalchemy.orm import Session

from sales_agent.features.knowledge.contracts import EmbeddingProvider, FactExtractor
from sales_agent.features.knowledge.standard_workflow import AnalyzeCallKnowledgeWorkflow


class AnalyzeCallKnowledgeLeanWorkflow(AnalyzeCallKnowledgeWorkflow):
    """Persist knowledge using fact-level speaker assignment and a lean extractor."""

    def __init__(
        self,
        session: Session,
        *,
        extractor: FactExtractor,
        embedder: EmbeddingProvider,
        full_document_direct_limit: int = 12_000,
    ) -> None:
        super().__init__(
            session,
            speaker_resolver=None,
            extractor=extractor,
            embedder=embedder,
            full_document_direct_limit=full_document_direct_limit,
            degrade_single_source_speaker_label=False,
        )
