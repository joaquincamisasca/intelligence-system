"""
Las 6 pruebas de punta a punta del sistema (la consigna pide al menos 5).

Cada prueba pasa por la API real (POST /tasks -> polling -> resultado) y por
lo tanto deja una traza completa en Arize Phoenix: supervisor, agentes,
llamadas al LLM y herramientas (incluida la búsqueda RAG).

  1. Pregunta documental (RAG puro)          -> solo investigador, con fuentes
  2. Pregunta TRAMPA (no está en los docs)   -> el sistema NO debe inventar
  3. Feedback de usuarios + análisis         -> investigador -> analista
  4. Cruce: documentación + feedback         -> RAG + comentarios + análisis
  5. Tarea crítica APROBADA (HITL)           -> pausa, aprobación humana, publicación
  6. Tarea crítica RECHAZADA (HITL)          -> pausa, rechazo humano, NO se publica

Uso (con el sistema levantado):
    python scripts/run_pruebas.py
Genera resultados/pruebas.md y resultados/pruebas.json.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PRUEBAS = [
    {
        "id": 1,
        "nombre": "Pregunta documental (RAG)",
        "query": "Según la documentación, ¿qué porcentaje mínimo de cobertura de tests se exige antes de mergear un pull request?",
        "espera": "Responde con el dato de 03_testing.md y lo cita en fuentes. No interviene el analista.",
    },
    {
        "id": 2,
        "nombre": "Pregunta trampa (no está en los documentos)",
        "query": "Según la documentación, ¿cuál es el presupuesto anual asignado al equipo de infraestructura?",
        "espera": "Admite que esa información no está en la documentación; no inventa una cifra.",
    },
    {
        "id": 3,
        "nombre": "Feedback de usuarios + análisis",
        "query": "Buscá los comentarios de los usuarios sobre el checkout, analizá el sentimiento general y calculá el rating promedio.",
        "espera": "Delegación investigador -> analista; sentimiento y promedio (3.2) calculados con herramientas.",
    },
    {
        "id": 4,
        "nombre": "Cruce documentación + feedback",
        "query": (
            "Según la documentación, ¿qué estrategia de despliegue se usa para los servicios críticos? "
            "Además, buscá los comentarios sobre el checkout, analizá el sentimiento y calculá el rating promedio."
        ),
        "espera": "Usa el RAG y la base de comentarios, y el analista procesa los comentarios.",
    },
    {
        "id": 5,
        "nombre": "Tarea crítica aprobada (HITL)",
        "query": "Buscá los comentarios sobre el checkout, analizá el sentimiento, calculá el rating promedio y publicá el reporte en el canal del equipo.",
        "espera": "Se pausa en WAITING_APPROVAL; tras la aprobación humana, se publica.",
        "decision": {"aprobado": True, "comentario": "Reporte revisado, OK para publicar."},
    },
    {
        "id": 6,
        "nombre": "Tarea crítica rechazada (HITL)",
        "query": "Buscá los comentarios sobre el onboarding, analizá el sentimiento y compartí el reporte con el equipo.",
        "espera": "Se pausa en WAITING_APPROVAL; tras el rechazo humano, NO se publica.",
        "decision": {"aprobado": False, "comentario": "Todavía faltan datos de otras features."},
    },
]


def esperar(cliente: httpx.Client, job_id: str, estados: set, timeout: float) -> dict:
    limite = time.time() + timeout
    while time.time() < limite:
        trabajo = cliente.get(f"/tasks/{job_id}").json()
        if trabajo["status"] in estados:
            return trabajo
        time.sleep(1)
    return {"status": "TIMEOUT", "error": f"No llegó a {estados} en {timeout}s"}


def correr(cliente: httpx.Client, prueba: dict, timeout: float) -> dict:
    inicio = time.perf_counter()
    creada = cliente.post("/tasks", json={"query": prueba["query"]}).json()
    job_id = creada["job_id"]
    print(f"[{prueba['id']}] {prueba['nombre']}  (job {job_id[:8]})")

    trabajo = esperar(cliente, job_id, {"DONE", "FAILED", "WAITING_APPROVAL"}, timeout)
    pausa = None

    if trabajo["status"] == "WAITING_APPROVAL" and "decision" in prueba:
        pausa = trabajo.get("interrupt")
        print(f"     pausa HITL -> decisión humana: aprobado={prueba['decision']['aprobado']}")
        cliente.post(f"/tasks/{job_id}/approve", json=prueba["decision"])
        trabajo = esperar(cliente, job_id, {"DONE", "FAILED"}, timeout)

    duracion = time.perf_counter() - inicio
    print(f"     -> {trabajo['status']} en {duracion:.1f}s")
    return {**prueba, "job_id": job_id, "estado": trabajo["status"], "duracion_s": round(duracion, 1), "pausa": pausa, "trabajo": trabajo}


def escribir_reporte(resultados: list, url: str, salida: Path) -> None:
    salida.mkdir(exist_ok=True)
    (salida / "pruebas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# Resultados de las pruebas de punta a punta",
        "",
        f"Fecha: {datetime.now().isoformat(timespec='seconds')} · API: {url}",
        "",
        "| # | Prueba | Estado | Duración | Fuentes citadas |",
        "|---|--------|--------|----------|-----------------|",
    ]
    for r in resultados:
        res = r["trabajo"].get("result") or {}
        lineas.append(f"| {r['id']} | {r['nombre']} | {r['estado']} | {r['duracion_s']} s | {', '.join(res.get('fuentes', [])) or '—'} |")

    for r in resultados:
        res = r["trabajo"].get("result") or {}
        lineas += ["", f"## {r['id']}. {r['nombre']}", "", f"**Consulta:** {r['query']}", "", f"**Qué se espera:** {r['espera']}", ""]
        lineas.append(f"**Estado:** {r['estado']} (job `{r['job_id']}`)")
        if r["estado"] != "DONE":
            lineas += ["", f"**Error:** {r['trabajo'].get('error')}"]
            continue
        lineas += ["", "**Delegación (traza):** " + " → ".join(t["agente"] for t in res.get("traza", []))]
        lineas += ["", f"**Respuesta:**", "", res.get("respuesta", ""), ""]
        if res.get("publicacion") is not None:
            lineas.append(f"**Publicación:** `{json.dumps(res['publicacion'], ensure_ascii=False)}`")

    (salida / "pruebas.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main(url: str, timeout: float) -> int:
    with httpx.Client(base_url=url, timeout=30) as cliente:
        try:
            salud = cliente.get("/health").json()
        except httpx.HTTPError as e:
            print(f"No se pudo conectar con {url}: {e}\n¿Está levantado el sistema? (docker compose up --build)")
            return 1
        print(f"Sistema OK: {salud}\n")

        resultados = [correr(cliente, p, timeout) for p in PRUEBAS]

    destino = Path(__file__).resolve().parent.parent / "resultados"
    escribir_reporte(resultados, url, destino)

    ok = sum(1 for r in resultados if r["estado"] == "DONE")
    print(f"\n{ok}/{len(resultados)} pruebas completadas. Reporte: {destino / 'pruebas.md'}")
    print("Ahora abrí Phoenix (http://localhost:6006) y sacá las capturas de las trazas.")
    return 0 if ok == len(resultados) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruebas de punta a punta contra la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=900, help="segundos máximos por prueba")
    args = parser.parse_args()
    sys.exit(main(args.url, args.timeout))
