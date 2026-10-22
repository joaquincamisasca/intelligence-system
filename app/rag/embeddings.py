"""
Embeddings ASYNC detrás de una interfaz única.

Regla del RAG (error #1 de la consigna del Módulo 3): se indexa y se
consulta con EXACTAMENTE el mismo modelo. Por eso hay un único punto de
creación (`get_embedder`) que usan tanto la ingesta como el recuperador.

Proveedores (variable EMBEDDINGS_PROVIDER):
- "fastembed": local, ONNX, sin PyTorch ni API key. Es el default, con un
  modelo MULTILINGÜE porque el corpus y las preguntas están en español.
- "openai": SDK asíncrono oficial (`AsyncOpenAI`). Requiere crédito.
- "hash": determinístico, sin descargas. SOLO para tests y CI offline.

fastembed es síncrono y consume CPU: se ejecuta en un thread con
`asyncio.to_thread`, así el event loop sigue atendiendo peticiones
mientras se calculan los vectores.
"""

import asyncio
import hashlib
import math
import re
import unicodedata
from functools import lru_cache
from typing import List, Protocol

from app.config import settings


class Embedder(Protocol):
    dimension: int

    async def embed_documents(self, textos: List[str]) -> List[List[float]]: ...

    async def embed_query(self, texto: str) -> List[float]: ...


class FastEmbedEmbedder:
    def __init__(self, modelo: str, dimension: int):
        self.modelo = modelo
        self.dimension = dimension
        self._modelo = None  # se carga recién al primer uso (descarga ~220 MB la primera vez)
        self._lock = asyncio.Lock()

    def _cargar(self):
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.modelo)

    def _embeber(self, textos: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._modelo.embed(textos)]

    async def _asegurar_modelo(self) -> None:
        async with self._lock:
            if self._modelo is None:
                self._modelo = await asyncio.to_thread(self._cargar)

    async def embed_documents(self, textos: List[str]) -> List[List[float]]:
        await self._asegurar_modelo()
        return await asyncio.to_thread(self._embeber, textos)

    async def embed_query(self, texto: str) -> List[float]:
        return (await self.embed_documents([texto]))[0]


class OpenAIEmbedder:
    def __init__(self, modelo: str, dimension: int, api_key: str):
        from openai import AsyncOpenAI  # cliente asíncrono oficial

        self.modelo = modelo
        self.dimension = dimension
        self._cliente = AsyncOpenAI(api_key=api_key)

    async def embed_documents(self, textos: List[str]) -> List[List[float]]:
        respuesta = await self._cliente.embeddings.create(model=self.modelo, input=textos)
        return [d.embedding for d in respuesta.data]

    async def embed_query(self, texto: str) -> List[float]:
        return (await self.embed_documents([texto]))[0]


def _sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


class HashEmbedder:
    """
    Bolsa de palabras "hasheada" a 384 dimensiones y normalizada. No es
    semántico, pero textos con palabras en común quedan cerca: alcanza
    para probar todo el pipeline (ingesta, upsert, query, fusión) sin
    descargar ningún modelo.
    """

    dimension = 384

    def _vector(self, texto: str) -> List[float]:
        v = [0.0] * self.dimension
        for palabra in re.findall(r"\w+", _sin_tildes(texto)):
            h = int(hashlib.md5(palabra.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    async def embed_documents(self, textos: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in textos]

    async def embed_query(self, texto: str) -> List[float]:
        return self._vector(texto)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    proveedor = settings.embeddings_provider
    if proveedor == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBEDDINGS_PROVIDER=openai requiere OPENAI_API_KEY en el .env")
        return OpenAIEmbedder(
            settings.openai_embedding_model,
            settings.embedding_dimension,
            settings.openai_api_key.get_secret_value(),
        )
    if proveedor == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(settings.embedding_model, settings.embedding_dimension)
