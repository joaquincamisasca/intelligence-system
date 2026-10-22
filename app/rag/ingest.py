"""
Ingesta: lee los documentos de DATA_DIR, los fragmenta y los deja listos
para indexar.

- El chunking cuenta TOKENS reales (tiktoken), no caracteres.
- La categoría sale del nombre del archivo ("02_seguridad.md" ->
  "seguridad"): agregar un documento nuevo no requiere tocar el código.
- Los IDs de chunk son determinísticos ("archivo-N"): re-ingestar el
  mismo contenido pisa los mismos registros en vez de duplicarlos.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

_encoding = None
_encoding_fallo = False


def _contar_tokens(texto: str) -> int:
    """
    Cuenta tokens reales con tiktoken. La primera vez tiktoken descarga su
    vocabulario (el Dockerfile lo deja precargado en la imagen). Si no hay
    red ni caché, se aproxima con 1 token ~ 4 caracteres en vez de romper
    la ingesta, y se avisa una sola vez.
    """
    global _encoding, _encoding_fallo
    if _encoding is None and not _encoding_fallo:
        try:
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # noqa: BLE001 - sin red y sin caché
            _encoding_fallo = True
            logger.warning(f"tiktoken no disponible ({type(e).__name__}); se aproxima con len/4.")
    if _encoding is None:
        return max(1, len(texto) // 4)
    return len(_encoding.encode(texto))


def _categoria(nombre_archivo: str) -> str:
    """'02_seguridad.md' -> 'seguridad'."""
    base = Path(nombre_archivo).stem
    return re.sub(r"^\d+_", "", base) or "general"


def cargar_documentos(data_dir: str | Path | None = None) -> List[Document]:
    carpeta = Path(data_dir or settings.data_dir)
    if not carpeta.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de documentos: {carpeta}")

    archivos = sorted([*carpeta.glob("*.md"), *carpeta.glob("*.txt")])
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .md/.txt en {carpeta}")

    documentos = []
    for archivo in archivos:
        documentos.append(
            Document(
                page_content=archivo.read_text(encoding="utf-8"),
                metadata={"source": archivo.name, "categoria": _categoria(archivo.name)},
            )
        )
    logger.info(f"Documentos cargados: {len(documentos)} desde {carpeta}")
    return documentos


def fragmentar_documentos(documentos: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        length_function=_contar_tokens,
    )
    chunks = splitter.split_documents(documentos)

    contador: dict = {}
    for chunk in chunks:
        fuente = chunk.metadata["source"]
        indice = contador.get(fuente, 0)
        chunk.metadata["chunk_index"] = indice
        chunk.metadata["id"] = f"{fuente}-{indice}"
        contador[fuente] = indice + 1

    logger.info(
        f"{len(chunks)} chunks (tamaño={settings.chunk_size_tokens} tokens, "
        f"solapamiento={settings.chunk_overlap_tokens})"
    )
    return chunks


async def preparar_chunks(data_dir: str | Path | None = None) -> List[Document]:
    """Lectura de disco + chunking en un thread (no bloquea el event loop)."""

    def _trabajo() -> List[Document]:
        return fragmentar_documentos(cargar_documentos(data_dir))

    return await asyncio.to_thread(_trabajo)
