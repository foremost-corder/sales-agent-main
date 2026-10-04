from sales_agent.integrations.embeddings.fastembed_provider import FastEmbedEmbeddingProvider
from sales_agent.core.config import Settings


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class FakeBackend:
    def embed(self, texts: list[str]):
        return iter([FakeVector([1.0] + [0.0] * 511) for _ in texts])


def test_fastembed_provider_returns_local_512d_vectors() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused:unused@127.0.0.1/unused",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-zh-v1.5",
        embedding_dimensions=512,
    )
    provider = FastEmbedEmbeddingProvider(settings)
    provider._backend = FakeBackend()  # type: ignore[assignment]

    vectors = provider.embed(["客户需要电脑", "客户预算十万元"])

    assert provider.provider_name == "fastembed"
    assert provider.model == "BAAI/bge-small-zh-v1.5"
    assert len(vectors) == 2
    assert all(len(vector) == 512 for vector in vectors)
