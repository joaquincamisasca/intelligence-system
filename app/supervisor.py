"""
Supervisor: el nodo router del orquestador (async).

En cada vuelta mira la conversación y lo que aportó cada especialista, y
decide con salida estructurada quién actúa a continuación — o si la
tarea ya está completa.

Robustez (lección práctica con modelos locales chicos como llama3.2, que
a veces NO devuelven la salida estructurada pedida):

1. Reintento: si el LLM no devuelve una decisión válida, se reintenta.
2. Plan B por reglas: si igual falla, la decisión se toma con reglas
   determinísticas en vez de romper el grafo.
3. Rúbrica de validación en código: pase lo que pase con el LLM, no se
   puede cerrar la tarea (FINISH) ni saltearse el orden hasta que los
   especialistas requeridos hayan aportado.

Ruteo dinámico: el investigador siempre actúa primero; el analista solo
es requerido si la solicitud pide análisis (sentimiento, promedio,
cálculo...). Una pregunta documental pura se resuelve con un solo
especialista, sin pasos de más.

Salvaguardas contra el "Supervisor Infinito": contador de pasos
(`MAX_PASOS_SUPERVISOR`) + criterio de Suficiencia explícito en el prompt.
"""

import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field

from app import evidencia
from app.config import settings
from app.hitl import _normalizar
from app.llm import get_llm  # noqa: F401  (los tests lo reemplazan en este módulo)
from app.state import OrquestadorState

logger = logging.getLogger(__name__)

# Cuántas veces se le pide al LLM una decisión válida antes del plan B.
INTENTOS_DECISION = 2

# Palabras (sin tildes) que indican que la solicitud pide un análisis.
_SEÑALES_DE_ANALISIS = ("analiz", "sentimiento", "promedio", "rating", "calcul", "estadistic")


def requiere_analisis(solicitud: str) -> bool:
    """¿La solicitud pide procesar datos (sentimiento, promedios, cálculos)? Regla determinística y auditable."""
    texto = _normalizar(solicitud)
    return any(señal in texto for señal in _SEÑALES_DE_ANALISIS)


def especialistas_requeridos(state: OrquestadorState) -> tuple:
    """Rúbrica: quiénes DEBEN contribuir (y en qué orden) antes de poder cerrar."""
    if requiere_analisis(_solicitud_original(state)):
        return ("investigador", "analista")
    return ("investigador",)


class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor: a quién delegar y con qué instrucción."""

    next_agent: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo nodo a ejecutar, o FINISH si la tarea ya está completa."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Instrucción puntual y acotada para el agente elegido (solo lo que "
            "necesita para hacer SU parte). Vacío si next_agent es FINISH."
        ),
    )
    razon: str = Field(default="", description="Justificación breve de la decisión, para trazabilidad.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "next_agent": "investigador",
                "instruccion": "Buscá en la documentación la cobertura mínima de tests.",
                "razon": "Todavía no se recolectó ningún dato.",
            }
        }
    )


PROMPT_SUPERVISOR = (
    "Sos el Supervisor de un equipo de dos agentes especialistas:\n"
    "- 'investigador': RECOLECTA información de dos fuentes: la documentación "
    "técnica de la empresa (búsqueda RAG) y los comentarios de usuarios. "
    "No analiza, solo busca y resume objetivamente.\n"
    "- 'analista': PROCESA información ya recolectada (sentimiento, "
    "estadísticas). Nunca sale a buscar datos.\n\n"
    "Dada la conversación y las contribuciones hasta ahora, decidí quién "
    "debe actuar a continuación, o si la tarea ya está completa (FINISH).\n\n"
    "Rúbrica de validación: primero actúa el investigador; el analista solo "
    "si la solicitud pide análisis o cálculos. No elijas FINISH hasta que "
    "hayan contribuido todos los especialistas requeridos.\n\n"
    "Criterio de Suficiencia (estricto): una vez que sus resultados responden "
    "la solicitud original, elegí FINISH — no pidas refinamientos por "
    "perfeccionismo.\n\n"
    "Contribuciones hasta ahora:\n{contribuciones}\n\n"
    "Si delegás, la 'instruccion' debe ser puntual: decile al agente "
    "exactamente qué necesitás (el agente NO ve el historial completo, solo "
    "tu instrucción)."
)

PROMPT_SINTESIS = (
    "Sintetizá la respuesta final para el usuario, en español, basándote "
    "ÚNICAMENTE en la EVIDENCIA y en las contribuciones de los especialistas.\n"
    "Reglas:\n"
    "1. Respondé directo, sin preámbulos como 'no tengo información' o 'no "
    "tengo acceso' cuando la evidencia sí contiene el dato.\n"
    "2. NO escribas ratings, promedios ni conteos de comentarios: el sistema "
    "los agrega verificados al final de tu respuesta. Sobre las opiniones de "
    "usuarios, resumí en 2 a 4 oraciones qué destacan y qué critican, sin "
    "listar los comentarios uno por uno. Podés nombrar el sentimiento general "
    "tal como figura en la evidencia. Los datos de la documentación (por "
    "ejemplo un porcentaje exigido) sí copialos tal cual.\n"
    "3. No atribuyas a la documentación nada que no esté en sus fragmentos.\n"
    "4. Cuando uses un dato de la documentación, citá el archivo entre "
    "corchetes, ej: [03_testing.md].\n"
    "5. Si la evidencia NO contiene lo que se pregunta, decí claramente que "
    "esa información no está en la documentación. Nunca la inventes.\n"
    "6. Respondé TODAS las partes de la pregunta.\n\n"
    "EVIDENCIA (lo que devolvieron las herramientas):\n{evidencia}\n\n"
    "Contribuciones de los especialistas:\n{contribuciones}"
)


def _formatear_contribuciones(contribuciones: dict) -> str:
    if not contribuciones:
        return "(ninguna todavía)"
    return "\n".join(f"- {agente}: {resumen}" for agente, resumen in contribuciones.items())


def _solicitud_original(state: OrquestadorState) -> str:
    """Texto del primer mensaje del usuario (la solicitud original)."""
    for mensaje in state["messages"]:
        if getattr(mensaje, "type", None) == "human":
            return mensaje.content
    return ""


def _instruccion_por_defecto(agente: str, state: OrquestadorState) -> str:
    """Instrucción puntual de respaldo, armada por reglas."""
    solicitud = _solicitud_original(state)
    contribuciones = state.get("contribuciones", {})

    if agente == "investigador":
        return (
            "Recolectá la información que necesita esta solicitud. Usá "
            "buscar_en_documentos para lo que esté en la documentación técnica "
            "y buscar_comentarios para opiniones de usuarios. Solicitud: "
            f"{solicitud}"
        )

    if agente == "analista":
        exactos = evidencia.datos_exactos_para_analista(state.get("evidencia") or {})
        datos = exactos or contribuciones.get("investigador", "(sin datos)")
        return (
            "Analizá el sentimiento general con analizar_sentimiento y calculá "
            "el rating promedio con calcular_estadisticas, usando estos datos.\n\n"
            f"Solicitud original: {solicitud}\n\n{datos}"
        )

    return ""


def _decision_por_reglas(state: OrquestadorState) -> DecisionSupervisor:
    """Plan B determinístico: el primer requerido que falta; si no falta nadie, se cierra."""
    contribuciones = state.get("contribuciones", {})

    for agente in especialistas_requeridos(state):
        if agente not in contribuciones:
            return DecisionSupervisor(
                next_agent=agente,
                instruccion=_instruccion_por_defecto(agente, state),
                razon=f"Decisión por reglas: '{agente}' todavía no contribuyó.",
            )

    return DecisionSupervisor(
        next_agent="FINISH", razon="Decisión por reglas: todos los especialistas requeridos ya contribuyeron."
    )


async def _decidir_con_llm(llm, mensajes: list) -> Optional[DecisionSupervisor]:
    """Pide una decisión estructurada (async) con reintentos. None si no obtuvo una válida."""
    cadena = llm.with_structured_output(DecisionSupervisor)

    for intento in range(1, INTENTOS_DECISION + 1):
        try:
            decision = await cadena.ainvoke(mensajes)
        except Exception as e:  # noqa: BLE001 - cualquier fallo del LLM cae al plan B
            logger.warning(f"Intento {intento}: el LLM falló al decidir ({type(e).__name__}: {e})")
            continue

        if isinstance(decision, DecisionSupervisor):
            return decision

        logger.warning(f"Intento {intento}: el LLM no devolvió una decisión estructurada válida.")

    return None


def _validar_decision(decision: DecisionSupervisor, state: OrquestadorState) -> DecisionSupervisor:
    """
    Rúbrica de validación (en código, independiente del LLM):
    1. Mientras falten especialistas requeridos, el próximo es el primero que falta.
    2. Al analista siempre se le pasan los datos del investigador.
    3. Si se delega, la instrucción no puede quedar vacía.
    """
    contribuciones = state.get("contribuciones", {})
    faltantes = [a for a in especialistas_requeridos(state) if a not in contribuciones]

    if faltantes and decision.next_agent != faltantes[0]:
        logger.warning(
            f"Decisión del LLM ('{decision.next_agent}') corregida a '{faltantes[0]}' por la rúbrica."
        )
        return DecisionSupervisor(
            next_agent=faltantes[0],
            instruccion=_instruccion_por_defecto(faltantes[0], state),
            razon=(
                f"Corregido por la rúbrica: primero debe actuar '{faltantes[0]}' "
                f"(el LLM había elegido '{decision.next_agent}')."
            ),
        )

    if not faltantes and decision.next_agent != "FINISH":
        # Ya contribuyeron todos los requeridos: no se re-delega (evita pedir refinamientos eternos).
        return DecisionSupervisor(
            next_agent="FINISH", razon="Corregido por la rúbrica: los especialistas requeridos ya contribuyeron."
        )

    if decision.next_agent == "FINISH":
        return decision

    instruccion = decision.instruccion.strip() or _instruccion_por_defecto(decision.next_agent, state)

    if decision.next_agent == "analista":
        # Al analista se le pasan las listas EXACTAS que devolvió la herramienta
        # (si existen), no el resumen en prosa del investigador.
        exactos = evidencia.datos_exactos_para_analista(state.get("evidencia") or {})
        datos = exactos or contribuciones.get("investigador")
        if datos and datos not in instruccion:
            instruccion += f"\n\n{datos}"

    return decision.model_copy(update={"instruccion": instruccion})


async def nodo_supervisor(state: OrquestadorState) -> dict:
    """Decide next_agent + instruccion, incrementa el contador y corta si se llegó a MAX_PASOS."""
    pasos = state.get("pasos", 0) + 1
    max_pasos = settings.max_pasos_supervisor

    if pasos > max_pasos:
        return {
            "next_agent": "FINISH",
            "pasos": pasos,
            "messages": [
                AIMessage(
                    content=(
                        f"Se alcanzó el máximo de {max_pasos} pasos sin que el Supervisor "
                        f"declarara la tarea completa. Cerrando con la información disponible."
                    ),
                    name="supervisor",
                )
            ],
        }

    llm = get_llm()
    contribuciones = state.get("contribuciones", {})

    prompt = PROMPT_SUPERVISOR.format(contribuciones=_formatear_contribuciones(contribuciones))
    mensajes = [{"role": "system", "content": prompt}, *state["messages"]]

    decision = await _decidir_con_llm(llm, mensajes)
    if decision is None:
        logger.warning("El LLM no logró decidir; se usa el plan B por reglas.")
        decision = _decision_por_reglas(state)

    decision = _validar_decision(decision, state)

    resultado = {
        "next_agent": decision.next_agent,
        "instruccion_para_agente": decision.instruccion,
        "pasos": pasos,
    }

    if decision.next_agent == "FINISH":
        ev = state.get("evidencia") or {}
        # Si hay cifras verificadas, la prosa del analista (que puede traer números mal
        # copiados) no se le pasa a la síntesis: alcanza con la evidencia.
        para_sintesis = dict(contribuciones)
        if ev.get("estadisticas") or ev.get("sentimiento"):
            para_sintesis.pop("analista", None)
        prompt_sintesis = PROMPT_SINTESIS.format(
            evidencia=evidencia.bloque_para_sintesis(ev),
            contribuciones=_formatear_contribuciones(para_sintesis),
        )
        sintesis = await llm.ainvoke(
            [
                {"role": "system", "content": prompt_sintesis},
                {"role": "user", "content": _solicitud_original(state)},
            ]
        )
        # Las cifras finales las agrega el CÓDIGO, no el LLM: el LLM redacta, el código calcula.
        fuentes = state.get("fuentes", [])
        texto = evidencia.limpiar_sintesis(sintesis.content, ev, fuentes) + evidencia.apendice_verificado(ev, fuentes)
        resultado["messages"] = [AIMessage(content=texto, name="supervisor")]
    else:
        resultado["messages"] = [
            AIMessage(
                content=f"Delego en '{decision.next_agent}'. Motivo: {decision.razon or 'sin detalle'}",
                name="supervisor",
            )
        ]

    return resultado
