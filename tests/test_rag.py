"""
Tests del módulo RAG con una base ChromaDB REAL en disco y embeddings
"hash" (sin descargar modelos): ingesta, idempotencia, recuperación
híbrida, fusión RRF y protecciones de configuración.
"""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import HashEmbedder
from app.rag.ingest import _categoria, cargar_documentos, fragmentar_documentos
from app.rag.retriever import RetrieverHibrido, tokenizar
from app.rag.schemas import Fragmento
from app.rag.service import RAGService
from app.rag.stores import ChromaStore, PineconeStore

GOLDEN = json.loads((Path(__file__).parent.parent / "data" / "golden_set.json").read_text(encoding="utf-8"))


@pytest.fixture
def chroma_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "chroma_collection", "test_coleccion")
    return tmp_path


# --- Ingesta ---

def test_categoria_sale_del_nombre_del_archivo():
    assert _categoria("02_seguridad.md") == "seguridad"
    assert _categoria("notas.txt") == "notas"


def test_cargar_y_fragmentar_documentos():
    chunks = fragmentar_documentos(cargar_documentos())
    assert len(chunks) >= 4
    assert {c.metadata["categoria"] for c in chunks} >= {"arquitectura", "seguridad", "testing", "deployment"}
    assert all(c.page_content.strip() for c in chunks)
    ids = [c.metadata["id"] for c in chunks]
    assert len(ids) == len(set(ids))  # IDs únicos y determinísticos


def test_carpeta_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_documentos(tmp_path / "no-existe")


# --- Servicio: ingesta idempotente ---

async def test_la_ingesta_es_idempotente(chroma_tmp):
    primero = RAGService()
    await primero.inicializar()
    cantidad = await primero._store.contar()
    assert cantidad >= 4

    # Un servicio nuevo (como tras reiniciar la API) NO duplica ni reindexa.
    segundo = RAGService()
    await segundo.inicializar()
    assert await segundo._store.contar() == cantidad


async def test_auto_ingest_desactivado_con_base_vacia_da_error(chroma_tmp, monkeypatch):
    monkeypatch.setattr(settings, "rag_auto_ingest", False)
    with pytest.raises(RuntimeError, match="RAG_AUTO_INGEST"):
        await RAGService().inicializar()


# --- Recuperación híbrida ---

@pytest.mark.parametrize("caso", GOLDEN, ids=[c["documento_id_esperado"] + ":" + c["pregunta"][:30] for c in GOLDEN])
async def test_el_documento_esperado_es_el_primero(chroma_tmp, caso):
    """Con el golden set de la pre-entrega 4: el documento correcto queda primero (Recall@1)."""
    servicio = RAGService()
    await servicio.inicializar()
    resultado = await servicio.buscar(caso["pregunta"], top_k=3)
    assert resultado.fuentes[0] == caso["documento_id_esperado"]


async def test_consulta_vacia_es_rechazada(chroma_tmp):
    servicio = RAGService()
    await servicio.inicializar()
    with pytest.raises(ValueError):
        await servicio._retriever.buscar("   ", 3)


def test_tokenizador_ignora_markdown_tildes_y_stopwords():
    assert tokenizar("**HashiCorp Vault**, ¿cómo se usa?") == ["hashicorp", "vault", "usa"]


class _StoreFalso:
    """Devuelve una lista fija: aísla la lógica de fusión del embedding y de la base."""

    nombre = "falso"

    def __init__(self, fragmentos):
        self._fragmentos = fragmentos

    async def buscar(self, vector, k):
        return self._fragmentos[:k]


def _frag(fuente, idx=0, score=0.9):
    return Fragmento(fuente=fuente, categoria="x", chunk_index=idx, texto=f"texto de {fuente}", score=score)


async def test_rrf_premia_lo_que_aparece_en_ambas_listas():
    chunks = [
        Document(page_content="kafka mensajería colas", metadata={"source": "a.md", "categoria": "x", "chunk_index": 0}),
        Document(page_content="vault secretos credenciales", metadata={"source": "b.md", "categoria": "x", "chunk_index": 0}),
        Document(page_content="otra cosa distinta", metadata={"source": "c.md", "categoria": "x", "chunk_index": 0}),
    ]
    # La búsqueda vectorial (falsa) pone primero a c.md y segundo a b.md;
    # BM25 pone primero a b.md (calza "vault"). b.md está en AMBAS listas: debe ganar.
    store = _StoreFalso([_frag("c.md"), _frag("b.md")])
    retriever = RetrieverHibrido(store, HashEmbedder(), chunks)

    resultado = await retriever.buscar("vault secretos", top_k=2)

    assert resultado[0].fuente == "b.md"
    assert resultado[0].score > resultado[1].score
    assert resultado[0].recuperadores == 2  # lo encontraron las dos búsquedas
    assert resultado[1].recuperadores == 1


async def test_bm25_sin_terminos_en_comun_no_devuelve_basura():
    chunks = [Document(page_content="kafka colas", metadata={"source": "a.md", "categoria": "x", "chunk_index": 0})]
    retriever = RetrieverHibrido(_StoreFalso([]), HashEmbedder(), chunks)
    assert retriever._buscar_bm25("xyzzy qwerty", 3) == []


# --- Protecciones de configuración ---

async def test_dimension_distinta_en_chroma_da_error_claro(chroma_tmp):
    await ChromaStore(settings.chroma_path, settings.chroma_collection).preparar(384)
    with pytest.raises(ValueError, match="dimensión"):
        await ChromaStore(settings.chroma_path, settings.chroma_collection).preparar(1536)


def test_pinecone_sin_api_key_da_error_claro(monkeypatch):
    monkeypatch.setattr(settings, "pinecone_api_key", None)
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        PineconeStore()


def test_el_nombre_del_indice_de_pinecone_sale_de_la_configuracion(monkeypatch):
    """Nada de IDs de índice escritos en el código (error común de la consigna)."""
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "pinecone_api_key", SecretStr("x"))
    monkeypatch.setattr(settings, "pinecone_index_name", "mi-indice-propio")
    assert PineconeStore().index_name == "mi-indice-propio"


def test_la_dimension_depende_del_proveedor_de_embeddings(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_provider", "openai")
    monkeypatch.setattr(settings, "openai_embedding_model", "text-embedding-3-small")
    assert settings.embedding_dimension == 1536
    monkeypatch.setattr(settings, "embeddings_provider", "hash")
    assert settings.embedding_dimension == 384


def test_las_rutas_relativas_se_resuelven_contra_la_raiz_del_proyecto():
    """Funciona igual aunque se ejecute desde otra carpeta (bug real visto en una corrida)."""
    from app.config import BASE_DIR, Settings

    s = Settings(data_dir="data/documentos", chroma_path="storage/chroma", sqlite_path="storage/x.sqlite")
    assert Path(s.data_dir) == BASE_DIR / "data" / "documentos"
    assert Path(s.chroma_path).is_absolute() and Path(s.sqlite_path).is_absolute()
