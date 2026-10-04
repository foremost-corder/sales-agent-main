"""Application-level embedding provider selection."""

from sales_agent.integrations.embeddings.fastembed_provider import FastEmbedEmbeddingProvider
from sales_agent.integrations.embeddings.openai_provider import OpenAIEmbeddingProvider
from sales_agent.core.config import Settings
from sales_agent.features.knowledge.contracts import EmbeddingProvider


class EmbeddingConfigurationError(ValueError):
    pass


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.casefold()
    if provider == "fastembed":
        return FastEmbedEmbeddingProvider(settings)
    if provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    raise EmbeddingConfigurationError(
        f"unsupported embedding provider: {settings.embedding_provider}"
    )
