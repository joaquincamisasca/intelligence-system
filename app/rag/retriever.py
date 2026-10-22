"""
Recuperador HÍBRIDO: búsqueda vectorial (semántica) + BM25 (léxica),
fusionadas con Reciprocal Rank Fusion (RRF) ponderada.

- La búsqueda vectorial entiende significado ("¿cómo guardan las
  contraseñas?" ~ "gestión de secretos") pero diluye términos exactos
  ("HashiCorp Vault", "testcontainers").
- BM25 hace lo contrario: es exacta con términos técnicos y ciega al
  significado.
- RRF combina las dos listas por POSICIÓN (no por puntaje, que viven en
  escalas distintas): score(doc) = Σ peso_i / (k + posición_i).

Ambas búsquedas corren en paralelo (`asyncio.gather`).
"""

import asyncio
import logging
import re
import unicodedata
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.rag.embeddings import Embedder
from app.rag.schemas import Fragmento
from app.rag.stores import VectorStore

logger = logging.getLogger(__name__)

# Palabras vacías del español: sin ellas BM25 puntúa por "de", "la", "que".
_STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "en", "a", "que", "se",
    "es", "por", "para", "con", "del", "al", "lo", "su", "sus", "como", "cual", "cuales", "qué",
    "que", "cuál", "cómo", "son", "ser", "esta", "este", "esto", "hay", "mas", "más", "si", "sí",
}


def tokenizar(texto: str) -> List[str]:
    """Minúsculas, sin tildes, solo palabras (evita que '**Vault**,' no calce con 'Vault')."""
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", sin_tildes) if t not in _STOPWORDS]


class RetrieverHibrido:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunks: List[Document],
        pesos: Tuple[float, float] = (0.5, 0.5),
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self.pesos = pesos
        self.rrf_k = rrf_k
        # El corpus de BM25 son los MISMOS chunks que se indexaron en la base vectorial.
        self._bm25 = BM25Okapi([tokenizar(c.page_content) for c in chunks])

    def _buscar_bm25(self, consulta: str, k: int) -> List[Fragmento]:
        tokens = tokenizar(consulta)
        if not tokens:
            return []
        puntajes = self._bm25.get_scores(tokens)
        orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)[:k]
        resultados = []
        for i in orden:
            if puntajes[i] <= 0:
                continue  # sin ningún término en común: no es un resultado
            c = self.chunks[i]
            resultados.append(
                Fragmento(
                    fuente=c.metadata["source"],
                    categoria=c.metadata["categoria"],
                    chunk_index=c.metadata["chunk_index"],
                    texto=c.page_content,
                    score=float(puntajes[i]),
                )
            )
        return resultados

    async def _buscar_vectorial(self, consulta: str, k: int) -> List[Fragmento]:
        vector = await self.embedder.embed_query(consulta)
        return await self.store.buscar(vector, k)

    async def buscar(self, consulta: str, top_k: int) -> List[Fragmento]:
        if not consulta or not consulta.strip():
            raise ValueError("La consulta no puede estar vacía.")

        # Se piden más candidatos que top_k a cada recuperador, para que la fusión tenga de dónde elegir.
        candidatos = max(top_k * 2, top_k + 2)
        vectoriales, lexicos = await asyncio.gather(
            self._buscar_vectorial(consulta, candidatos),
            asyncio.to_thread(self._buscar_bm25, consulta, candidatos),
        )

        puntaje: Dict[str, float] = {}
        por_id: Dict[str, Fragmento] = {}
        apariciones: Dict[str, int] = {}
        for peso, lista in zip(self.pesos, (vectoriales, lexicos)):
            for posicion, frag in enumerate(lista, start=1):
                clave = f"{frag.fuente}-{frag.chunk_index}"
                puntaje[clave] = puntaje.get(clave, 0.0) + peso / (self.rrf_k + posicion)
                apariciones[clave] = apariciones.get(clave, 0) + 1
                por_id.setdefault(clave, frag)

        ordenados = sorted(puntaje, key=puntaje.get, reverse=True)[:top_k]
        resultado = [
            por_id[c].model_copy(update={"score": round(puntaje[c], 5), "recuperadores": apariciones[c]})
            for c in ordenados
        ]
        logger.info(
            f"RAG híbrido: '{consulta[:60]}' -> {len(vectoriales)} vectoriales + "
            f"{len(lexicos)} léxicos -> {len(resultado)} fusionados"
        )
        return resultado
