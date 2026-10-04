"""CPU-local embeddings backed by FastEmbed and ONNX Runtime."""

import logging
from pathlib import Path
from time import perf_counter

from fastembed import TextEmbedding

from sales_agent.core.config import Settings


logger = logging.getLogger(__name__)


class FastEmbedEmbeddingProvider:
    provider_name = "fastembed"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._cache_dir = Path(settings.embedding_cache_dir)
        self._threads = settings.embedding_threads or None
        self._backend: TextEmbedding | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started = perf_counter()
        logger.info(
            "local_embedding_start model=%s dimensions=%d text_count=%d total_characters=%d",
            self.model,
            self.dimensions,
            len(texts),
            sum(len(text) for text in texts),
        )
        vectors = [vector.tolist() for vector in self._model().embed(texts)]
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions for vector in vectors
        ):
            actual = len(vectors[0]) if vectors else 0
            raise ValueError(
                f"local embedding model returned {actual} dimensions; "
                f"configured EMBEDDING_DIMENSIONS={self.dimensions}"
            )
        logger.info(
            "local_embedding_completed model=%s vector_count=%d elapsed_ms=%d",
            self.model,
            len(vectors),
            round((perf_counter() - started) * 1000),
        )
        return vectors

    def _model(self) -> TextEmbedding:
        if self._backend is None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._backend = TextEmbedding(
                model_name=self.model,
                cache_dir=str(self._cache_dir),
                threads=self._threads,
            )
        return self._backend
