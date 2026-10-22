"""
Configuración centralizada y VALIDADA del sistema.

Todo se lee de variables de entorno (o del archivo .env) con
pydantic-settings: si una variable tiene un valor inválido (por ejemplo,
CHECKPOINTER=postgres), el sistema falla al arrancar con un mensaje
claro, no a mitad de una ejecución.

Ningún ID de índice, nombre de modelo ni secreto está escrito en el
código: todo vive acá y se sobreescribe desde el .env.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Raíz del proyecto: las rutas relativas del .env se resuelven contra ella, así
# el sistema funciona igual sin importar desde qué carpeta se lo ejecute.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Redis: cola, estado de trabajos y checkpoints de LangGraph ---
    redis_url: str = "redis://localhost:6379"
    # "redis": AsyncRedisSaver (requiere Redis Stack). Si no está disponible
    #          se cae a "sqlite" con un aviso fuerte (ver app/checkpointing.py).
    # "sqlite": AsyncSqliteSaver en un archivo (también sobrevive reinicios).
    # "memory": solo para pruebas; el estado se pierde al reiniciar.
    checkpointer: Literal["redis", "sqlite", "memory"] = "redis"
    sqlite_path: str = "./storage/checkpoints.sqlite"

    # --- Workers y límites ---
    worker_concurrency: int = Field(5, ge=1, le=50)
    job_ttl_seconds: int = Field(60 * 60 * 24, ge=60)
    recursion_limit: int = Field(25, ge=5)
    max_pasos_supervisor: int = Field(6, ge=2)

    # --- LLM ---
    generation_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    # Ventana de contexto de Ollama. Su default es chico y corta en silencio los
    # prompts largos (síntesis con varias contribuciones): se amplía.
    ollama_num_ctx: int = Field(8192, ge=2048)
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-3-5-haiku-latest"
    anthropic_api_key: Optional[SecretStr] = None

    # --- RAG: base vectorial ---
    # "chroma" (default): local y persistente, sin cuentas ni API keys.
    # "pinecone": índice Serverless en la nube (requiere PINECONE_API_KEY).
    vector_store: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "documentos_tecnicos"
    pinecone_api_key: Optional[SecretStr] = None
    pinecone_index_name: str = "intelligence-system"
    pinecone_namespace: str = "documentos-tecnicos"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- RAG: embeddings ---
    # "fastembed": local (ONNX, sin PyTorch ni API key). "openai": de pago.
    # "hash": determinístico y sin descargas; SOLO para tests y CI offline.
    embeddings_provider: Literal["fastembed", "openai", "hash"] = "fastembed"
    # Multilingüe (el corpus y las preguntas están en español). 384 dimensiones.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- RAG: ingesta y recuperación ---
    data_dir: str = "./data/documentos"
    chunk_size_tokens: int = Field(650, ge=100, le=2000)
    chunk_overlap_tokens: int = Field(100, ge=0)
    rag_top_k: int = Field(3, ge=1, le=10)
    rag_auto_ingest: bool = True

    # --- Observabilidad (Arize Phoenix) ---
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "intelligence-system"

    @field_validator("sqlite_path", "chroma_path", "data_dir")
    @classmethod
    def _ruta_absoluta(cls, valor: str) -> str:
        ruta = Path(valor)
        return str(ruta if ruta.is_absolute() else (BASE_DIR / ruta).resolve())

    @property
    def embedding_dimension(self) -> int:
        """Dimensión del vector según el proveedor (el índice se crea con este valor)."""
        if self.embeddings_provider == "openai":
            return 1536 if "small" in self.openai_embedding_model else 3072
        if self.embeddings_provider == "hash":
            return 384
        if "multilingual-e5-large" in self.embedding_model:
            return 1024
        if "mpnet" in self.embedding_model:
            return 768
        return 384


settings = Settings()
