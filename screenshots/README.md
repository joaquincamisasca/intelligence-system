# Capturas de Arize Phoenix

Evidencia de la observabilidad, tomada del dashboard de Phoenix
(http://localhost:6006, proyecto `intelligence-system`) después de correr
`scripts/run_pruebas.py` (6 pruebas) y `scripts/load_test.py` (5 tareas concurrentes),
con el sistema levantado con `docker compose up --build` y llama3.2 en Ollama.

| Archivo | Qué muestra |
|---|---|
| `01_lista_de_trazas.png` | Lista de trazas: 1 traza por ejecución (`job.ejecucion`), con latencia, tokens y costo de cada una. |
| `02_arbol_de_spans_rag.png` | Prueba 1 (pregunta documental): árbol de spans `job.ejecucion → LangGraph → supervisor → investigador → buscar_en_documentos`. A la derecha, la consulta que armó el agente y los fragmentos que devolvió el RAG (`03_testing.md`, "cobertura mínima del 80%"). |
| `03a_investigador_y_rubrica_del_supervisor.png` | Prueba 3: el investigador con sus herramientas (`buscar_comentarios`, `buscar_en_documentos`). En el input se ve la rúbrica del Supervisor en acción: "Corregido por la rúbrica: primero debe actuar 'investigador' (el LLM había elegido 'analista')". |
| `03b_analista_y_sus_herramientas.png` | Prueba 3: el analista llamando a `analizar_sentimiento` y `calcular_estadisticas` con los ratings exactos `[5, 2, 3, 5, 1]` → promedio 3.2. |
| `04a_hitl_pausa_con_interrupt.png` | Prueba 5: el grafo se pausa en `aprobacion_humana` con `interrupt()` (`app/hitl.py`). Phoenix lo registra como evento de tipo excepción, porque así implementa LangGraph la pausa: no es un fallo. |
| `04b_hitl_reanudacion_y_publicacion.png` | Prueba 5, segunda traza de la misma sesión: tras la aprobación humana, el grafo se reanuda desde el checkpoint y publica (`"publicado": true`, con el comentario del revisor). |
| `05_metricas_latencia_y_costo.png` | Pestaña Metrics: volumen de trazas, percentiles de latencia (P50 a P99.9) y costo estimado por modelo (`llama3.2`). |
| `06_resumen_stats_del_proyecto.png` | Panel Stats del proyecto: total de trazas y latencias P50/P99. |

Notas:
- **Costo:** Ollama corre local (costo real $0). Se registró en Phoenix un precio de referencia para `llama3.2` (`scripts/setup_phoenix_pricing.py`), así el dashboard estima cuánto costaría la misma carga en la nube. Total: US$ 0,013 por 217.190 tokens, ~US$ 0,0003 por ejecución (`scripts/phoenix_metrics.py`).
- **Barras rojas ("error") en los gráficos:** corresponden a las pausas de aprobación humana (el `interrupt()` de LangGraph se registra como excepción) y no a fallas: las 11 ejecuciones de pruebas y carga terminaron en `DONE`.
- **Latencias altas (P99 de ~6,5 min):** son 4 tareas de una prueba de carga anterior a la corrección de la ventana de contexto de Ollama. Con 5 tareas concurrentes, Ollama atiende de a una, así que hacen fila. La prueba de carga final dio p50 111 s y p95 124 s (`resultados/load_test.json`); las tareas sueltas tardan entre 18 y 63 s.
