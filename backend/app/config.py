"""Hanna backend configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ChromaDB settings removed — migrated to PostgreSQL+pgvector

    # Embedding
    embedding_backend: str = "bge-m3"  # "bge-m3" or "openai"
    bge_m3_url: str = "http://host.docker.internal:8104"  # search instance
    bge_m3_ingest_url: str = "http://host.docker.internal:8114"  # ingest instance
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"  # OpenAI fallback
    embedding_dimensions: int = 1024  # BGE-M3: 1024, OpenAI: 1536

    # Cohere (reranking fallback)
    cohere_api_key: str = ""
    rerank_model: str = "rerank-v3.5"

    # Microsoft Graph API
    graph_tenant_id: str = ""
    graph_client_id: str = ""
    graph_client_secret: str = ""
    graph_user_email: str = ""  # Hanna's mailbox for auth
    shared_mailboxes: str = "info@neuzrt.hu,lakossagienergetika@neuzrt.hu"

    # Chunking
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 100  # tokens

    # Search
    search_top_k: int = 20  # initial retrieval per method (semantic + BM25)
    rerank_top_k: int = 5  # final results after reranking

    # Contextual Compression (post-rerank noise filtering)
    compression_enabled: bool = True
    compression_score_floor: float = 0.01  # remove chunks with rerank_score below this
    compression_gap_ratio: float = 0.35    # cut if score drops to <35% of previous
    compression_min_results: int = 3       # always keep at least this many

    # Adaptive k (dynamic retrieval depth by query complexity)
    adaptive_k_enabled: bool = True

    # HyDE
    hyde_enabled: bool = True
    hyde_model: str = "gpt-4o-mini"
    hyde_timeout: float = 3.0
    hyde_max_tokens: int = 350

    answer_model: str = "gpt-4o-mini"
    answer_max_tokens: int = 1500
    answer_temperature: float = 0.1

    class Config:
        env_prefix = ""


settings = Settings()
