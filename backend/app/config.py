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

    # Anthropic (Claude — fallback 1)
    anthropic_api_key: str = ""

    # Google (Gemini — fallback 2)
    google_api_key: str = ""

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
    compression_score_floor: float = 0.005  # remove chunks with rerank_score below this
    compression_gap_ratio: float = 0.08    # cut if score < 8% of TOP result (very conservative)
    compression_min_results: int = 5       # always keep at least this many

    # Adaptive k (dynamic retrieval depth by query complexity)
    adaptive_k_enabled: bool = True

    # HyDE
    hyde_enabled: bool = False  # Disabled: domain-aware query expansion supersedes HyDE
    hyde_model: str = "gpt-4o-mini"
    hyde_timeout: float = 8.0
    hyde_max_tokens: int = 350

    answer_model: str = "gpt-4o-mini"
    answer_max_tokens: int = 1500
    answer_temperature: float = 0.1

    # Discord notifications (optional)
    discord_webhook_url: str = ""      # Webhook URL (if available)
    discord_bot_token: str = ""        # Bot token (alternative to webhook)
    discord_channel_id: str = ""       # Channel ID for bot messages

    # Report output directory (Obsidian !reports or any writable path)
    report_dir: str = "/app/data/reports"

    # Autonomous processing
    auto_process_enabled: bool = False  # Feature flag for autonomous email processing

    # Feedback analytics (Level 1 learning)
    feedback_analytics_enabled: bool = True
    langfuse_dataset_name: str = "hanna-draft-pairs"

    # DSPy prompt optimization (Level 3 learning)
    dspy_enabled: bool = False
    dspy_min_training_pairs: int = 30

    # Gap detection (Level 4 learning)
    gap_detection_min_cluster: int = 3
    gap_detection_similarity_threshold: float = 0.7

    # Authority snapshots
    authority_snapshot_dir: str = "/app/data/authority_snapshots"

    # OETP MySQL database (readonly)
    oetp_db_host: str = "185.187.73.44"
    oetp_db_port: int = 3307
    oetp_db_user: str = "tarolo_readonly"
    oetp_db_password: str = ""       # Set in .env: OETP_DB_PASSWORD
    oetp_db_name: str = "tarolo_neuzrt_hu_db"
    oetp_db_enabled: bool = False    # Feature flag — set to true when password configured

    class Config:
        env_prefix = ""


settings = Settings()
