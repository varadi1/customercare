"""CustomerCare backend configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Embedding
    embedding_backend: str = "bge-m3"  # "bge-m3" or "openai"
    bge_m3_url: str = "http://host.docker.internal:8104"  # search instance
    bge_m3_ingest_url: str = "http://host.docker.internal:8114"  # ingest instance
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"  # OpenAI fallback
    embedding_dimensions: int = 1024  # BGE-M3: 1024, OpenAI: 1536

    # Anthropic (Claude — primary LLM)
    anthropic_api_key: str = ""

    # Google (Gemini — fallback)
    google_api_key: str = ""

    # Cohere (reranking fallback)
    cohere_api_key: str = ""
    rerank_model: str = "rerank-v3.5"

    # Microsoft Graph API
    graph_tenant_id: str = ""
    graph_client_id: str = ""
    graph_client_secret: str = ""
    graph_user_email: str = ""
    shared_mailboxes: str = ""  # Comma-separated, set in .env or program.yaml

    # Chunking
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 100  # tokens

    # Search
    search_top_k: int = 20
    rerank_top_k: int = 5

    # Contextual Compression
    compression_enabled: bool = True
    compression_score_floor: float = 0.005
    compression_gap_ratio: float = 0.08
    compression_min_results: int = 5

    # Adaptive k
    adaptive_k_enabled: bool = True

    # HyDE
    hyde_enabled: bool = False
    hyde_model: str = "gpt-4o-mini"
    hyde_timeout: float = 8.0
    hyde_max_tokens: int = 350

    answer_model: str = "gpt-4o-mini"
    answer_max_tokens: int = 1500
    answer_temperature: float = 0.1

    # Discord notifications (optional)
    discord_webhook_url: str = ""
    discord_bot_token: str = ""
    discord_channel_id: str = ""

    # Report output
    report_dir: str = "/app/data/reports"

    # Autonomous processing
    auto_process_enabled: bool = False

    # Feedback analytics (Level 1 learning)
    feedback_analytics_enabled: bool = True
    langfuse_dataset_name: str = "customercare-draft-pairs"

    # DSPy prompt optimization (Level 3 learning)
    dspy_enabled: bool = False
    dspy_min_training_pairs: int = 30

    # Gap detection (Level 4 learning)
    gap_detection_min_cluster: int = 3
    gap_detection_similarity_threshold: float = 0.7

    # Authority snapshots
    authority_snapshot_dir: str = "/app/data/authority_snapshots"

    # Program-specific external database (optional, set in .env)
    program_db_host: str = ""
    program_db_port: int = 3306
    program_db_user: str = ""
    program_db_password: str = ""
    program_db_name: str = ""
    program_db_enabled: bool = False

    # Program config file path
    program_config: str = "config/program.yaml"

    class Config:
        env_prefix = ""


settings = Settings()
