"""
Agente Especialista de Análisis.

Recibe datos YA RECOLECTADOS y los procesa con sus herramientas (análisis
de sentimiento y estadística). Nunca sale a buscar información por su
cuenta. Herramientas async, con entrada y salida validadas por Pydantic.
"""

from langchain_core.tools import tool

from app.agents.schemas import (
    AnalizarSentimientoInput,
    CalcularEstadisticasInput,
    ResultadoEstadisticas,
    ResultadoSentimiento,
)

# Heurística determinística (conteo de palabras clave), a propósito en vez
# de un LLM: el resultado es reproducible y testeable con asserts exactos.
_PALABRAS_POSITIVAS = {
    "encantó", "excelente", "rápido", "recomiendo", "bien", "buena",
    "buenísimo", "genial", "claro",
}
_PALABRAS_NEGATIVAS = {
    "tarda", "frustrante", "error", "mala", "lento", "confuso",
    "confusa", "perdí", "problema",
}


@tool(args_schema=AnalizarSentimientoInput)
async def analizar_sentimiento(comentarios: list[str]) -> dict:
    """
    Clasifica cada comentario como positivo, negativo o neutro según las
    palabras clave que contiene, y devuelve el sentimiento general.
    """
    positivos = negativos = neutros = 0

    for comentario in comentarios:
        texto = comentario.lower()
        tiene_positiva = any(p in texto for p in _PALABRAS_POSITIVAS)
        tiene_negativa = any(n in texto for n in _PALABRAS_NEGATIVAS)

        if tiene_positiva and not tiene_negativa:
            positivos += 1
        elif tiene_negativa and not tiene_positiva:
            negativos += 1
        else:
            neutros += 1

    if positivos > negativos:
        general = "positivo"
    elif negativos > positivos:
        general = "negativo"
    else:
        general = "mixto"

    return ResultadoSentimiento(
        positivos=positivos, negativos=negativos, neutros=neutros, sentimiento_general=general
    ).model_dump()


@tool(args_schema=CalcularEstadisticasInput)
async def calcular_estadisticas(numeros: list[float]) -> dict:
    """Calcula promedio, mínimo, máximo y cantidad de una lista de números (ej: ratings)."""
    return ResultadoEstadisticas(
        promedio=round(sum(numeros) / len(numeros), 2),
        minimo=min(numeros),
        maximo=max(numeros),
        cantidad=len(numeros),
    ).model_dump()


TOOLS_ANALISTA = [analizar_sentimiento, calcular_estadisticas]

PROMPT_ANALISTA = (
    "Sos el Agente de Análisis. Recibís datos ya recolectados (por "
    "ejemplo, comentarios de usuarios con sus ratings) y tu trabajo es "
    "PROCESARLOS con tus herramientas: analizar_sentimiento y "
    "calcular_estadisticas. Nunca inventes datos que no te hayan sido "
    "provistos explícitamente en la instrucción — si te falta "
    "información para calcular algo, decilo en vez de asumir valores."
)


def crear_agente_analista(llm):
    """Sub-agente ReAct (LangChain 1.x `create_agent`). Se invoca con `ainvoke`."""
    from langchain.agents import create_agent

    return create_agent(llm, tools=TOOLS_ANALISTA, system_prompt=PROMPT_ANALISTA)
