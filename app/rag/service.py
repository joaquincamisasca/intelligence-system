"""
Servicio RAG: une embeddings + base vectorial + recuperador híbrido y se
inicializa UNA vez (al arrancar la API).

La ingesta es idempotente: si la base ya tiene vectores no se reindexa
(ahorra tiempo y costo); si está vacía, se indexa sola. Para forzar un
reindexado, `python -m scripts.ingest --forzar`.
"""

import asyncio
import logging
from typing import Optional

from app.config import settings
from app.rag.embeddings import get_embedder
from app.rag.ingest import preparar_chunks
from app.rag.retriever import RetrieverHibrido
from app.rag.schemas import ResultadoRAG
from app.rag.stores import crear_store

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self._retriever: Optional[RetrieverHibrido] = None
        self._store = None
        self._lock = asyncio.Lock()

    @property
    def listo(self) -> bool:
        return self._retriever is not None

    async def inicializar(self, forzar_reindexado: bool = False) -> None:
        async with self._lock:
            if self._retriever is not None and not forzar_reindexado:
                return

            embedder = get_embedder()
            store = crear_store()
            await store.preparar(embedder.dimension)

            chunks = await preparar_chunks()
            existentes = await store.contar()

            if existentes == 0 or forzar_reindexado:
                if not settings.rag_auto_ingest and not forzar_reindexado:
                    raise RuntimeError(
                        "La base vectorial está vacía y RAG_AUTO_INGEST=false. "
                        "Ejecutá `python -m scripts.ingest` primero."
                    )
                logger.info(f"Indexando {len(chunks)} chunks en {store.nombre}...")
                vectores = await embedder.embed_documents([c.page_content for c in chunks])
                await store.upsert(chunks, vectores)
                logger.info("Ingesta completa.")
            else:
                logger.info(f"Base vectorial ({store.nombre}) ya poblada con {existentes} vectores: no se reindexa.")

            self._store = store
            self._retriever = RetrieverHibrido(store, embedder, chunks)

    async def buscar(self, consulta: str, top_k: Optional[int] = None) -> ResultadoRAG:
        if self._retriever is None:
            await self.inicializar()
        k = top_k or settings.rag_top_k
        fragmentos = await self._retriever.buscar(consulta, k)

        fuentes: list = []
        for f in fragmentos:
            if f.fuente not in fuentes:
                fuentes.append(f.fuente)

        return ResultadoRAG(consulta=consulta, fragmentos=fragmentos, total=len(fragmentos), fuentes=fuentes)

    @property
    def store_nombre(self) -> str:
        return self._store.nombre if self._store else "no inicializado"


_servicio: Optional[RAGService] = None


def get_rag() -> RAGService:
    global _servicio
    if _servicio is None:
        _servicio = RAGService()
    return _servicio


def reset_rag() -> None:
    """Solo para tests: descarta el servicio y el embedder cacheados."""
    global _servicio
    _servicio = None
    get_embedder.cache_clear()
