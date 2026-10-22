"""
Checkpointer de LangGraph: dónde se guarda el estado de cada ejecución.

Es lo que hace que una conversación NO se pierda cuando el servidor se
reinicia: el grafo guarda su estado (mensajes, contribuciones, punto de
pausa del HITL) en cada paso, indexado por `thread_id` (= job_id).

Modos (variable CHECKPOINTER):
- "redis" (default): `AsyncRedisSaver`. Requiere Redis Stack (módulos
  RediSearch + RedisJSON). Si el Redis no los trae, NO se cae la API:
  se avisa fuerte y se usa SQLite, que también persiste a disco.
- "sqlite": `AsyncSqliteSaver` sobre un archivo (volumen de Docker).
- "memory": `InMemorySaver`, solo para pruebas (se pierde al reiniciar).
"""

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


async def _abrir_redis(stack: AsyncExitStack):
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    saver = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(settings.redis_url))
    # Lectura de prueba: si faltan los módulos de Redis Stack, falla ACÁ y no en el primer trabajo.
    await saver.aget_tuple({"configurable": {"thread_id": "__healthcheck__", "checkpoint_ns": ""}})
    return saver


async def _abrir_sqlite(stack: AsyncExitStack):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(settings.sqlite_path))


@asynccontextmanager
async def abrir_checkpointer():
    """Context manager async que entrega el checkpointer configurado (con fallback a SQLite)."""
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning("CHECKPOINTER=memory: el estado del grafo NO sobrevive a un reinicio.")
        yield InMemorySaver()
        return

    async with AsyncExitStack() as stack:
        saver = None

        if settings.checkpointer == "redis":
            try:
                saver = await _abrir_redis(stack)
                logger.info("Checkpointer de LangGraph: AsyncRedisSaver (Redis Stack).")
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"No se pudo usar AsyncRedisSaver ({type(e).__name__}: {e}). "
                    f"¿Redis sin módulos de Redis Stack? Se usa SQLite como respaldo persistente."
                )

        if saver is None:
            saver = await _abrir_sqlite(stack)
            logger.info(f"Checkpointer de LangGraph: AsyncSqliteSaver ({settings.sqlite_path}).")

        yield saver
