"""
Bases vectoriales intercambiables (ChromaDB local o Pinecone Serverless)
detrás de la misma interfaz async.

Los SDKs de ambas son síncronos: cada llamada se ejecuta con
`asyncio.to_thread`, así NINGUNA operación de base vectorial bloquea el
event loop (el sistema sigue atendiendo peticiones concurrentes mientras
Chroma o Pinecone responden).

Cuál se usa lo decide VECTOR_STORE en el .env; el resto del sistema no se
entera.
"""

import asyncio
import logging
import time
from typing import List, Protocol

from langchain_core.documents import Document

from app.config import settings
from app.rag.schemas import Fragmento

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    nombre: str

    async def preparar(self, dimension: int) -> None: ...

    async def contar(self) -> int: ...

    async def upsert(self, chunks: List[Document], vectores: List[List[float]]) -> None: ...

    async def buscar(self, vector: List[float], k: int) -> List[Fragmento]: ...


# ---------------------------------------------------------------------------
# ChromaDB (local, persistente en disco)
# ---------------------------------------------------------------------------

class ChromaStore:
    nombre = "chroma"

    def __init__(self, path: str, coleccion: str):
        self.path = path
        self.nombre_coleccion = coleccion
        self._coleccion = None

    def _preparar(self, dimension: int) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        cliente = chromadb.PersistentClient(
            path=self.path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        coleccion = cliente.get_or_create_collection(
            name=self.nombre_coleccion,
            metadata={"hnsw:space": "cosine", "dimension": dimension},
        )
        dimension_guardada = (coleccion.metadata or {}).get("dimension")
        if dimension_guardada is not None and int(dimension_guardada) != dimension:
            raise ValueError(
                f"La colección '{self.nombre_coleccion}' fue creada con dimensión "
                f"{dimension_guardada}, pero los embeddings configurados tienen "
                f"dimensión {dimension}. Cambiá CHROMA_COLLECTION o borrá la carpeta "
                f"{self.path} para reindexar (indexar y consultar con modelos de "
                f"embeddings distintos da resultados sin sentido)."
            )
        self._coleccion = coleccion

    async def preparar(self, dimension: int) -> None:
        await asyncio.to_thread(self._preparar, dimension)

    async def contar(self) -> int:
        return await asyncio.to_thread(self._coleccion.count)

    def _upsert(self, chunks: List[Document], vectores: List[List[float]]) -> None:
        self._coleccion.upsert(
            ids=[c.metadata["id"] for c in chunks],
            embeddings=vectores,
            documents=[c.page_content for c in chunks],
            metadatas=[
                {
                    "source": c.metadata["source"],
                    "categoria": c.metadata["categoria"],
                    "chunk_index": c.metadata["chunk_index"],
                }
                for c in chunks
            ],
        )

    async def upsert(self, chunks: List[Document], vectores: List[List[float]]) -> None:
        await asyncio.to_thread(self._upsert, chunks, vectores)

    def _buscar(self, vector: List[float], k: int) -> List[Fragmento]:
        total = self._coleccion.count()
        if total == 0:
            return []
        res = self._coleccion.query(query_embeddings=[vector], n_results=min(k, total))
        fragmentos = []
        for texto, meta, distancia in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            fragmentos.append(
                Fragmento(
                    fuente=meta["source"],
                    categoria=meta["categoria"],
                    chunk_index=meta["chunk_index"],
                    texto=texto,
                    score=1.0 - float(distancia),  # distancia coseno -> similitud
                )
            )
        return fragmentos

    async def buscar(self, vector: List[float], k: int) -> List[Fragmento]:
        return await asyncio.to_thread(self._buscar, vector, k)


# ---------------------------------------------------------------------------
# Pinecone Serverless (nube)
# ---------------------------------------------------------------------------

class PineconeStore:
    nombre = "pinecone"

    def __init__(self):
        if not settings.pinecone_api_key:
            raise ValueError(
                "VECTOR_STORE=pinecone requiere PINECONE_API_KEY en el .env "
                "(plan Serverless gratuito en https://app.pinecone.io/)."
            )
        self.index_name = settings.pinecone_index_name
        self.namespace = settings.pinecone_namespace
        self._pc = None
        self._index = None

    def _preparar(self, dimension: int) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

        if self._pc.has_index(self.index_name):
            info = self._pc.describe_index(self.index_name)
            if info.dimension != dimension:
                raise ValueError(
                    f"El índice '{self.index_name}' tiene dimensión {info.dimension} pero los "
                    f"embeddings configurados tienen dimensión {dimension}. Cambiá "
                    f"PINECONE_INDEX_NAME para usar un índice nuevo."
                )
        else:
            logger.info(f"Creando índice Serverless '{self.index_name}' (dimensión={dimension})...")
            self._pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
            while not self._pc.describe_index(self.index_name).status["ready"]:
                time.sleep(2)

        self._index = self._pc.Index(self.index_name)

    async def preparar(self, dimension: int) -> None:
        await asyncio.to_thread(self._preparar, dimension)

    def _contar(self) -> int:
        stats = self._index.describe_index_stats()
        namespaces = stats["namespaces"] if "namespaces" in stats else {}
        info = namespaces.get(self.namespace)
        if not info:
            return 0
        return int(info["vector_count"] if "vector_count" in info else info.vector_count)

    async def contar(self) -> int:
        return await asyncio.to_thread(self._contar)

    def _upsert(self, chunks: List[Document], vectores: List[List[float]]) -> None:
        registros = [
            {
                "id": c.metadata["id"],
                "values": v,
                "metadata": {
                    "text": c.page_content,  # el texto viaja en la metadata: sin base adicional
                    "source": c.metadata["source"],
                    "categoria": c.metadata["categoria"],
                    "chunk_index": c.metadata["chunk_index"],
                },
            }
            for c, v in zip(chunks, vectores)
        ]
        for inicio in range(0, len(registros), 100):
            self._index.upsert(vectors=registros[inicio : inicio + 100], namespace=self.namespace)

    async def upsert(self, chunks: List[Document], vectores: List[List[float]]) -> None:
        await asyncio.to_thread(self._upsert, chunks, vectores)

    def _buscar(self, vector: List[float], k: int) -> List[Fragmento]:
        res = self._index.query(vector=vector, top_k=k, namespace=self.namespace, include_metadata=True)
        fragmentos = []
        for m in res.matches:
            meta = dict(m.metadata or {})
            fragmentos.append(
                Fragmento(
                    fuente=meta.get("source", "desconocida"),
                    categoria=meta.get("categoria", "general"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    texto=meta.get("text", "") or "(sin texto)",
                    score=float(m.score),
                )
            )
        return fragmentos

    async def buscar(self, vector: List[float], k: int) -> List[Fragmento]:
        return await asyncio.to_thread(self._buscar, vector, k)


def crear_store() -> VectorStore:
    if settings.vector_store == "pinecone":
        return PineconeStore()
    return ChromaStore(settings.chroma_path, settings.chroma_collection)
