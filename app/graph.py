"""
Grafo del orquestador multi-agente: Supervisor + especialistas + nodo de
aprobación humana. Todos los nodos son async.

    START -> supervisor --(next_agent)--> investigador -> supervisor
                 |                        analista     -> supervisor
                 +--> FINISH --(¿tarea crítica?)--> aprobacion_humana -> END
                                     |                (interrupt: pausa)
                                     +--- no -------------------------> END

- `investigador` usa el RAG híbrido como herramienta (buscar_en_documentos).
- `build_graph(checkpointer)` recibe el checkpointer desde afuera: en
  producción Redis (o SQLite), en los tests InMemorySaver. Es lo que
  permite que el estado sobreviva a un reinicio y que el flujo HITL
  pueda pausarse y retomarse.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app import evidencia, supervisor
from app.agents.analyst_agent import crear_agente_analista
from app.agents.research_agent import crear_agente_investigador
from app.hitl import NOMBRE_NODO, es_tarea_critica, nodo_aprobacion_humana
from app.state import OrquestadorState
from app.supervisor import _solicitud_original, nodo_supervisor

logger = logging.getLogger(__name__)

# Techo de pasos de un sub-agente (llamar herramienta -> responder). Evita que un
# modelo local chico entre en un bucle de herramientas.
RECURSION_LIMIT_AGENTE = 12


async def _ejecutar_agente(agente, instruccion: str) -> dict:
    """Invoca un sub-agente con SOLO la instrucción puntual (async) y devuelve su resultado."""
    try:
        return await agente.ainvoke(
            {"messages": [HumanMessage(content=instruccion)]},
            config={"recursion_limit": RECURSION_LIMIT_AGENTE},
        )
    except GraphRecursionError:
        logger.warning("Un sub-agente superó su límite de pasos; se devuelve un resultado parcial.")
        return {"messages": [AIMessage(content="El agente no logró terminar dentro de su límite de pasos.")]}


async def _nodo_investigador(state: OrquestadorState) -> dict:
    agente = crear_agente_investigador(supervisor.get_llm())
    resultado = await _ejecutar_agente(agente, state["instruccion_para_agente"])
    respuesta = resultado["messages"][-1].content

    # Lo que DEVOLVIERON las herramientas se guarda aparte de lo que dijo el agente.
    acumulada = evidencia.fusionar(state.get("evidencia"), evidencia.extraer(resultado["messages"]))

    fuentes = list(state.get("fuentes", []))
    for f in evidencia.fuentes_relevantes(acumulada, respuesta, _solicitud_original(state)):
        if f not in fuentes:
            fuentes.append(f)

    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": {**state.get("contribuciones", {}), "investigador": respuesta},
        "fuentes": fuentes,
        "evidencia": acumulada,
    }


async def _nodo_analista(state: OrquestadorState) -> dict:
    agente = crear_agente_analista(supervisor.get_llm())
    resultado = await _ejecutar_agente(agente, state["instruccion_para_agente"])
    respuesta = resultado["messages"][-1].content

    # Verificación: las cifras se recalculan con las mismas herramientas sobre los
    # datos recuperados. Si el LLM le pasó números mal copiados a sus herramientas,
    # los datos verificados no se ven afectados.
    actual = state.get("evidencia") or {}
    verificado = await evidencia.verificar_analisis(actual)

    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": {**state.get("contribuciones", {}), "analista": respuesta},
        "evidencia": evidencia.fusionar(actual, verificado),
    }


def _enrutar(state: OrquestadorState) -> str:
    """Arista condicional desde el Supervisor. Si cierra y la tarea es crítica, pasa por la aprobación humana."""
    destino = state["next_agent"]
    if destino != "FINISH":
        return destino
    if es_tarea_critica(_solicitud_original(state)):
        return NOMBRE_NODO
    return END


def build_graph(checkpointer=None):
    """Arma y compila el grafo (checkpointer obligatorio para HITL y para persistencia)."""
    builder = StateGraph(OrquestadorState)

    builder.add_node("supervisor", nodo_supervisor)
    builder.add_node("investigador", _nodo_investigador)
    builder.add_node("analista", _nodo_analista)
    builder.add_node(NOMBRE_NODO, nodo_aprobacion_humana)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _enrutar,
        {"investigador": "investigador", "analista": "analista", NOMBRE_NODO: NOMBRE_NODO, END: END},
    )
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    builder.add_edge(NOMBRE_NODO, END)

    return builder.compile(checkpointer=checkpointer)
