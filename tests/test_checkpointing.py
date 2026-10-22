"""
Tests de PERSISTENCIA: el estado del grafo sobrevive a un "reinicio del servidor".

Se simula el reinicio abriendo el checkpointer, pausando el grafo en la
aprobación humana, CERRANDO todo, y volviendo a abrir uno nuevo sobre el
mismo archivo: la ejecución debe seguir pausada y poder reanudarse.
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app import checkpointing
from app.config import settings
from app.graph import build_graph
from app.hitl import NOMBRE_NODO

CRITICA = "Analizá los comentarios del checkout y publicá el reporte en el canal del equipo."


def _entrada(texto):
    return {"messages": [HumanMessage(content=texto)], "contribuciones": {}, "pasos": 0, "fuentes": [], "publicacion": None}


@pytest.fixture
def sqlite_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "checkpointer", "sqlite")
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "estado.sqlite"))


async def test_una_ejecucion_pausada_sobrevive_al_reinicio(modelo_falso, sqlite_tmp):
    cfg = {"configurable": {"thread_id": "job-persistente"}}

    # --- Primer "arranque": corre hasta la pausa y se apaga ---
    async with checkpointing.abrir_checkpointer() as saver:
        grafo = build_graph(saver)
        await grafo.ainvoke(_entrada(CRITICA), cfg)
        assert (await grafo.aget_state(cfg)).next == (NOMBRE_NODO,)

    # --- Segundo "arranque": objetos nuevos, mismo archivo ---
    async with checkpointing.abrir_checkpointer() as saver:
        grafo = build_graph(saver)
        snapshot = await grafo.aget_state(cfg)

        assert snapshot.next == (NOMBRE_NODO,)  # seguía pausada
        assert set(snapshot.values["contribuciones"]) == {"investigador", "analista"}  # y con su estado

        await grafo.ainvoke(Command(resume={"aprobado": True, "comentario": "tras reinicio"}), cfg)
        final = await grafo.aget_state(cfg)

        assert final.next == ()
        assert final.values["publicacion"]["publicado"] is True


async def test_los_hilos_distintos_no_se_mezclan(modelo_falso, sqlite_tmp):
    async with checkpointing.abrir_checkpointer() as saver:
        grafo = build_graph(saver)
        await grafo.ainvoke(_entrada(CRITICA), {"configurable": {"thread_id": "A"}})

        otro = await grafo.aget_state({"configurable": {"thread_id": "B"}})
        assert not otro.values


async def test_si_redis_no_tiene_los_modulos_se_cae_a_sqlite(tmp_path, monkeypatch):
    """Redis inalcanzable (o sin Redis Stack): la API no se cae, usa SQLite y lo avisa."""
    monkeypatch.setattr(settings, "checkpointer", "redis")
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1")  # puerto sin servicio
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "respaldo.sqlite"))

    async with checkpointing.abrir_checkpointer() as saver:
        assert type(saver).__name__ == "AsyncSqliteSaver"


async def test_modo_memory_entrega_inmemorysaver(monkeypatch):
    monkeypatch.setattr(settings, "checkpointer", "memory")
    async with checkpointing.abrir_checkpointer() as saver:
        assert type(saver).__name__ == "InMemorySaver"


def test_un_valor_invalido_de_checkpointer_falla_al_arrancar():
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(checkpointer="postgres")
