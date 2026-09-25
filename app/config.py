from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ai_gateway_semantic_cache"
    upstream_base_url: str = ""
    upstream_api_key: str = ""
    upstream_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    semantic_direct_threshold: float = 0.97
    semantic_validate_threshold: float = 0.90
    default_max_input_tokens: int = 800
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

settings = Settings()
