"""Hanna backend configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    chroma_collection: str = "hanna_knowledge"

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Cohere (reranking)
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

    class Config:
        env_prefix = ""


settings = Settings()
