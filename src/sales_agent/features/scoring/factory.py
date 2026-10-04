"""Application composition for the otherwise transport-neutral scoring workflow."""

from sqlalchemy.orm import Session

from sales_agent.integrations.embeddings.factory import build_embedding_provider
from sales_agent.core.config import get_settings
from sales_agent.features.scoring.model_agents import (
    OpenAIScoreReviewAgent,
    OpenAISectionScoringAgent,
)
from sales_agent.features.scoring.policy import SALES_KEY_BEHAVIOR_POLICY_V1
from sales_agent.features.scoring.retrieval import PgVectorScoreEvidenceRetriever
from sales_agent.features.scoring.workflow import SalesPerformanceScoringWorkflow


def build_sales_performance_scoring_workflow(
    session: Session,
) -> SalesPerformanceScoringWorkflow:
    settings = get_settings()
    retriever = PgVectorScoreEvidenceRetriever(
        session, embedder=build_embedding_provider(settings)
    )
    # Separate instances make each section an independently replaceable scoring
    # agent while preserving one shared contract and policy.
    section_agents = {
        section.section_id: OpenAISectionScoringAgent(settings)
        for section in SALES_KEY_BEHAVIOR_POLICY_V1.sections
    }
    return SalesPerformanceScoringWorkflow(
        retriever=retriever,
        section_agents=section_agents,
        reviewer=OpenAIScoreReviewAgent(settings),
    )
