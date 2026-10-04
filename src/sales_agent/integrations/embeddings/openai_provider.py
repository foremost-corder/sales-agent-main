import logging
from time import perf_counter

from openai import OpenAI

from sales_agent.agent.chat_model import ChatModelNotConfiguredError, ChatModelRequestError
from sales_agent.core.config import Settings


logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._settings.has_embedding_api_key:
            raise ChatModelNotConfiguredError("EMBEDDING_API_KEY is not configured")
        started = perf_counter()
        logger.info(
            "embedding_request model=%s dimensions=%d text_count=%d total_characters=%d",
            self.model,
            self.dimensions,
            len(texts),
            sum(len(text) for text in texts),
        )
        key = self._settings.embedding_api_key_value
        assert key is not None
        options: dict[str, object] = {
            "api_key": key,
            "max_retries": 1,
            "timeout": 30.0,
        }
        base_url = self._settings.embedding_base_url or self._settings.openai_base_url
        if base_url:
            options["base_url"] = base_url
        try:
            response = OpenAI(**options).embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
        except Exception as exc:
            raise ChatModelRequestError(
                "embedding request failed",
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
            ) from exc
        vectors = [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]
        if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("embedding provider returned an unexpected shape")
        logger.info(
            "embedding_completed model=%s vector_count=%d elapsed_ms=%d",
            self.model,
            len(vectors),
            round((perf_counter() - started) * 1000),
        )
        return vectors
