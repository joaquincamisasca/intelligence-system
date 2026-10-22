"""
API REST (FastAPI) del Intelligence System.

Endpoints:
- POST /tasks                    encola la tarea y devuelve job_id al instante (202)
- GET  /tasks/{job_id}           estado y resultado (polling)
- POST /tasks/{job_id}/approve   decisión humana para tareas críticas (HITL)
- GET  /tasks/{job_id}/checkpoint  qué hay guardado en el checkpointer de esa ejecución
- POST /rag/search               consulta directa al RAG híbrido (sin agentes)
- GET  /health                   chequeo de Redis, checkpointer, base vectorial y LLM

Ningún endpoint ejecuta el grafo ni llama a un LLM: solo leen/escriben en
Redis con el cliente asíncrono. El trabajo pesado lo hacen los workers en
segundo plano, así el event loop nunca se bloquea.

Documentación interactiva: http://localhost:8000/docs

Para levantarla:  uvicorn app.main:app   (o `docker compose up --build`)
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request, status

from app.checkpointing import abrir_checkpointer
from app.config import settings
from app.graph import build_graph
from app.hitl import es_tarea_critica
from app.job_store import WAITING_APPROVAL, JobStore
from app.observability import init_observability
from app.rag.schemas import ResultadoRAG
from app.rag.service import get_rag
from app.schemas import (
    Aprobacion,
    AprobacionAceptada,
    BusquedaRAG,
    CheckpointEstado,
    Salud,
    TareaCreada,
    TareaEntrada,
    Trabajo,
)
from app.worker import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _crear_redis_por_defecto():
    from redis.asyncio import Redis

    return Redis.from_url(settings.redis_url, decode_responses=True)


def _crear_fastapi(lifespan) -> FastAPI:
    """
    Crea la app SIN trazas HTTP automáticas.

    Las versiones recientes de FastAPI emiten un span por cada request
    cuando hay un tracer global (que Phoenix configura). Los GET de polling
    se volverían trazas propias de ~4 ms y contaminarían el p95 y el costo
    por ejecución. Queremos 1 traza = 1 trabajo, así que se apaga ese tracing.
    """
    opciones = dict(
        title="Intelligence System — API",
        description=(
            "Sistema de IA de punta a punta: RAG híbrido + orquestador multi-agente "
            "(Supervisor) con checkpointer, aprobación humana y trazas en Arize Phoenix."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    try:
        return FastAPI(**opciones, telemetry={"tracing": False})
    except TypeError:  # versiones de FastAPI sin esa opción (y sin esos spans)
        return FastAPI(**opciones)


def create_app(
    redis_factory: Optional[Callable] = None,
    checkpointer_factory: Optional[Callable] = None,
    concurrencia: int = settings.worker_concurrency,
) -> FastAPI:
    redis_factory = redis_factory or _crear_redis_por_defecto
    checkpointer_factory = checkpointer_factory or abrir_checkpointer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_observability()
        redis = redis_factory()
        store = JobStore(redis, ttl_segundos=settings.job_ttl_seconds)

        # El RAG se inicializa al arrancar (e indexa solo si la base está vacía):
        # si está mal configurado, se ve ACÁ y no en la primera tarea.
        rag = get_rag()
        await rag.inicializar()

        async with checkpointer_factory() as checkpointer:
            grafo = build_graph(checkpointer)
            workers = WorkerPool(store, grafo, concurrencia)
            workers.iniciar()

            app.state.store = store
            app.state.redis = redis
            app.state.grafo = grafo
            app.state.checkpointer_nombre = type(checkpointer).__name__
            try:
                yield
            finally:
                await workers.detener()

        await redis.aclose()

    app = _crear_fastapi(lifespan)

    @app.post("/tasks", response_model=TareaCreada, status_code=status.HTTP_202_ACCEPTED)
    async def crear_tarea(entrada: TareaEntrada, request: Request) -> TareaCreada:
        """Encola la tarea y devuelve el job_id inmediatamente (no espera al grafo)."""
        store: JobStore = request.app.state.store
        job_id = str(uuid.uuid4())
        critica = es_tarea_critica(entrada.query)

        await store.crear(job_id, entrada.query, requiere_aprobacion=critica)
        await store.encolar({"tipo": "run", "job_id": job_id})

        logger.info(f"Tarea {job_id} encolada (crítica={critica}).")
        return TareaCreada(job_id=job_id, status="PENDING", requiere_aprobacion=critica)

    @app.get("/tasks/{job_id}", response_model=Trabajo, response_model_exclude_none=True)
    async def obtener_tarea(job_id: str, request: Request) -> dict:
        """Estado actual del trabajo. Pensado para polling."""
        trabajo = await request.app.state.store.obtener(job_id)
        if trabajo is None:
            raise HTTPException(status_code=404, detail=f"No existe el trabajo {job_id}.")
        return trabajo

    @app.post("/tasks/{job_id}/approve", response_model=AprobacionAceptada, status_code=status.HTTP_202_ACCEPTED)
    async def aprobar_tarea(job_id: str, aprobacion: Aprobacion, request: Request) -> AprobacionAceptada:
        """Decisión humana sobre una tarea pausada (WAITING_APPROVAL). Encola la reanudación y vuelve al instante."""
        store: JobStore = request.app.state.store
        trabajo = await store.obtener(job_id)
        if trabajo is None:
            raise HTTPException(status_code=404, detail=f"No existe el trabajo {job_id}.")
        if trabajo["status"] != WAITING_APPROVAL:
            raise HTTPException(
                status_code=409,
                detail=f"El trabajo está en estado {trabajo['status']}, no espera aprobación.",
            )

        # Se marca RUNNING antes de encolar: un doble click no encola dos reanudaciones.
        await store.actualizar(job_id, status="RUNNING")
        await store.encolar({"tipo": "resume", "job_id": job_id, "decision": aprobacion.model_dump()})

        logger.info(f"Tarea {job_id}: decisión humana recibida (aprobado={aprobacion.aprobado}).")
        return AprobacionAceptada(job_id=job_id, status="RUNNING", aprobado=aprobacion.aprobado)

    @app.get("/tasks/{job_id}/checkpoint", response_model=CheckpointEstado)
    async def ver_checkpoint(job_id: str, request: Request) -> CheckpointEstado:
        """
        Lo que el CHECKPOINTER tiene guardado de esta ejecución (estado del grafo).
        Sirve para demostrar la persistencia: tras reiniciar la API, una tarea
        pausada sigue acá, guardada en Redis/SQLite y no en la memoria del proceso.
        """
        cfg = {"configurable": {"thread_id": job_id}}
        snapshot = await request.app.state.grafo.aget_state(cfg)
        valores = snapshot.values or {}
        return CheckpointEstado(
            job_id=job_id,
            guardado=bool(valores),
            checkpointer=request.app.state.checkpointer_nombre,
            siguiente_nodo=list(snapshot.next),
            pasos_supervisor=valores.get("pasos", 0),
            mensajes=len(valores.get("messages", [])),
            contribuciones=list(valores.get("contribuciones", {}).keys()),
        )

    @app.post("/rag/search", response_model=ResultadoRAG)
    async def buscar_rag(entrada: BusquedaRAG) -> ResultadoRAG:
        """Consulta directa al RAG híbrido (BM25 + vectorial), sin pasar por los agentes."""
        return await get_rag().buscar(entrada.consulta, entrada.top_k)

    @app.get("/health", response_model=Salud)
    async def health(request: Request) -> Salud:
        try:
            await request.app.state.redis.ping()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Redis no disponible: {e}")
        return Salud(
            status="ok",
            redis="ok",
            checkpointer=request.app.state.checkpointer_nombre,
            vector_store=get_rag().store_nombre,
            llm=f"{settings.generation_provider}",
        )

    return app


app = create_app()
