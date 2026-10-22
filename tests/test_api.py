"""
Tests de la API completa: FastAPI real + workers reales + grafo real + RAG
real (Chroma, embeddings "hash"), con Redis simulado (fakeredis),
InMemorySaver y el LLM falso.
"""

import time
from contextlib import asynccontextmanager

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.main import create_app

CHECKOUT = "Analizá el feedback del checkout"
CRITICA = "Analizá el checkout y publicá el reporte"
DOCUMENTAL = "Según la documentación, ¿qué cobertura mínima de tests se exige?"


@asynccontextmanager
async def _checkpointer_en_memoria():
    yield InMemorySaver()


@pytest.fixture
def cliente(modelo_falso):
    app = create_app(
        redis_factory=lambda: fakeredis.aioredis.FakeRedis(decode_responses=True),
        checkpointer_factory=_checkpointer_en_memoria,
        concurrencia=5,
    )
    with TestClient(app) as c:
        yield c


def _esperar_estado(cliente, job_id, estados, timeout=30):
    limite = time.time() + timeout
    trabajo = {}
    while time.time() < limite:
        trabajo = cliente.get(f"/tasks/{job_id}").json()
        if trabajo["status"] in estados:
            return trabajo
        time.sleep(0.05)
    raise AssertionError(f"El trabajo {job_id} no llegó a {estados}: quedó en {trabajo.get('status')}")


def test_health(cliente):
    cuerpo = cliente.get("/health").json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["redis"] == "ok"
    assert cuerpo["checkpointer"] == "InMemorySaver"
    assert cuerpo["vector_store"] == "chroma"


def test_post_devuelve_job_id_al_instante(cliente):
    inicio = time.perf_counter()
    r = cliente.post("/tasks", json={"query": CHECKOUT})
    demora = time.perf_counter() - inicio

    assert r.status_code == 202
    assert r.json()["status"] == "PENDING"
    assert r.json()["job_id"]
    assert demora < 1.0  # no espera a que corra el grafo


def test_flujo_completo_hasta_done(cliente):
    job_id = cliente.post("/tasks", json={"query": CHECKOUT}).json()["job_id"]
    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})

    assert trabajo["status"] == "DONE"
    assert "Síntesis" in trabajo["result"]["respuesta"]
    assert set(trabajo["result"]["contribuciones"]) == {"investigador", "analista"}
    assert trabajo["duracion_s"] >= 0


def test_las_cifras_verificadas_ganan_aunque_el_llm_diga_otra_cosa(cliente):
    """El analista falso dice 'promedio 2.67'; los 5 comentarios reales dan 3.2, y eso es lo que se informa."""
    job_id = cliente.post("/tasks", json={"query": CHECKOUT}).json()["job_id"]
    resultado = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})["result"]

    assert resultado["datos_verificados"]["estadisticas"]["promedio"] == 3.2
    assert resultado["datos_verificados"]["cantidad_comentarios"] == 5
    assert "Rating promedio: 3.2" in resultado["respuesta"]


def test_pregunta_documental_usa_el_rag_y_devuelve_las_fuentes(cliente):
    """El investigador llama a buscar_en_documentos; las fuentes quedan en el resultado y NO pasa por el analista."""
    job_id = cliente.post("/tasks", json={"query": DOCUMENTAL}).json()["job_id"]
    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})

    assert trabajo["status"] == "DONE", trabajo.get("error")
    assert trabajo["result"]["fuentes"] == ["03_testing.md"]  # solo el relevante, no los 4
    assert set(trabajo["result"]["contribuciones"]) == {"investigador"}


def test_cinco_peticiones_concurrentes_terminan_todas(cliente):
    ids = [cliente.post("/tasks", json={"query": f"{CHECKOUT} #{i}"}).json()["job_id"] for i in range(5)]
    assert len(set(ids)) == 5

    estados = [_esperar_estado(cliente, j, {"DONE", "FAILED"})["status"] for j in ids]
    assert estados == ["DONE"] * 5


def test_flujo_hitl_aprobado(cliente):
    r = cliente.post("/tasks", json={"query": CRITICA})
    assert r.json()["requiere_aprobacion"] is True
    job_id = r.json()["job_id"]

    trabajo = _esperar_estado(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
    assert trabajo["status"] == "WAITING_APPROVAL"
    assert trabajo["interrupt"]["accion"] == "publicar_reporte"

    r = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "OK"})
    assert r.status_code == 202
    assert r.json() == {"job_id": job_id, "status": "RUNNING", "aprobado": True}

    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})
    assert trabajo["status"] == "DONE"
    assert trabajo["result"]["publicacion"]["publicado"] is True


def test_flujo_hitl_rechazado(cliente):
    job_id = cliente.post("/tasks", json={"query": CRITICA}).json()["job_id"]
    _esperar_estado(cliente, job_id, {"WAITING_APPROVAL"})

    cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "No"})
    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})

    assert trabajo["result"]["publicacion"]["publicado"] is False


def test_endpoint_checkpoint_muestra_el_estado_guardado(cliente):
    job_id = cliente.post("/tasks", json={"query": CRITICA}).json()["job_id"]
    _esperar_estado(cliente, job_id, {"WAITING_APPROVAL"})

    pausado = cliente.get(f"/tasks/{job_id}/checkpoint").json()
    assert pausado["guardado"] is True
    assert pausado["siguiente_nodo"] == ["aprobacion_humana"]
    assert pausado["contribuciones"] == ["investigador", "analista"]

    cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    _esperar_estado(cliente, job_id, {"DONE"})
    assert cliente.get(f"/tasks/{job_id}/checkpoint").json()["siguiente_nodo"] == []


def test_checkpoint_de_un_trabajo_inexistente_no_esta_guardado(cliente):
    assert cliente.get("/tasks/no-existe/checkpoint").json()["guardado"] is False


def test_rag_search_directo(cliente):
    r = cliente.post("/rag/search", json={"consulta": "¿Cómo se gestionan los secretos y credenciales?", "top_k": 2})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["total"] == 2
    assert cuerpo["fuentes"][0] == "02_seguridad.md"


def test_aprobar_tarea_que_no_espera_aprobacion_da_409(cliente):
    job_id = cliente.post("/tasks", json={"query": CHECKOUT}).json()["job_id"]
    _esperar_estado(cliente, job_id, {"DONE"})

    assert cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True}).status_code == 409


def test_trabajo_inexistente_da_404(cliente):
    assert cliente.get("/tasks/no-existe").status_code == 404
    assert cliente.post("/tasks/no-existe/approve", json={"aprobado": True}).status_code == 404


# --- Validación Pydantic de las entradas de la API ---

def test_query_vacia_da_422(cliente):
    assert cliente.post("/tasks", json={"query": ""}).status_code == 422


def test_query_demasiado_larga_da_422(cliente):
    assert cliente.post("/tasks", json={"query": "a" * 2001}).status_code == 422


def test_body_sin_query_da_422(cliente):
    assert cliente.post("/tasks", json={}).status_code == 422


def test_aprobacion_con_tipo_invalido_da_422(cliente):
    job_id = cliente.post("/tasks", json={"query": CRITICA}).json()["job_id"]
    assert cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": "quizás"}).status_code == 422


def test_rag_search_con_top_k_fuera_de_rango_da_422(cliente):
    assert cliente.post("/rag/search", json={"consulta": "cobertura", "top_k": 50}).status_code == 422


def test_la_documentacion_openapi_expone_todos_los_endpoints(cliente):
    rutas = cliente.get("/openapi.json").json()["paths"]
    for ruta in ("/tasks", "/tasks/{job_id}", "/tasks/{job_id}/approve", "/tasks/{job_id}/checkpoint", "/rag/search", "/health"):
        assert ruta in rutas
