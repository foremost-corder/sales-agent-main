from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Sales Agent"
    environment: str = "development"
    database_url: str = Field(min_length=1)
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-5-mini"
    scoring_model: str | None = None
    scoring_max_output_tokens: int = Field(default=8192, ge=512, le=16384)
    scoring_timeout_seconds: float = Field(default=180.0, ge=30.0, le=600.0)
    scoring_max_retries: int = Field(default=3, ge=0, le=10)
    scoring_retry_base_seconds: float = Field(default=1.0, ge=0.0, le=30.0)
    scoring_retry_max_seconds: float = Field(default=10.0, ge=0.1, le=120.0)
    chat_max_output_tokens: int = Field(default=1200, ge=1, le=8192)
    conversation_context_messages: int = Field(default=12, ge=2, le=100)
    agent_max_steps: int = Field(default=6, ge=1, le=12)
    agent_final_answer_max_repairs: int = Field(default=3, ge=0, le=3)
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimensions: int = Field(default=512, ge=1)
    embedding_cache_dir: str = ".models/fastembed"
    embedding_threads: int = Field(default=0, ge=0, le=64)
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    fact_extraction_max_output_tokens: int = Field(default=4096, ge=256, le=4096)
    knowledge_max_repairs: int = Field(default=3, ge=0, le=3)
    knowledge_evaluator_model: str | None = None
    knowledge_evaluation_max_output_tokens: int = Field(default=2400, ge=256, le=4096)

    @property
    def has_openai_api_key(self) -> bool:
        if self.openai_api_key is None:
            return False
        value = self.openai_api_key.get_secret_value().strip()
        return bool(value and not value.startswith("replace-with"))

    @property
    def has_embedding_api_key(self) -> bool:
        return self.embedding_api_key_value is not None

    @property
    def knowledge_evaluator_model_name(self) -> str:
        return self.knowledge_evaluator_model or self.chat_model

    @property
    def scoring_model_name(self) -> str:
        return self.scoring_model or self.chat_model

    @property
    def embedding_api_key_value(self) -> str | None:
        for key in (self.embedding_api_key, self.openai_api_key):
            if key is None:
                continue
            value = key.get_secret_value().strip()
            if value and not value.startswith("replace-with"):
                return value
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
