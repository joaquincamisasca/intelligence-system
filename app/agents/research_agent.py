"""
Agente Especialista de Investigación.

Su responsabilidad es RECOLECTAR información — nunca analizarla (eso es
del Agente de Análisis). Tiene dos herramientas, ambas async y validadas
con Pydantic:

1. `buscar_en_documentos`: el RAG híbrido (BM25 + vectorial) sobre la
   documentación técnica. ES el RAG de los módulos 3 y 4, convertido en
   herramienta del agente.
2. `buscar_comentarios`: base (simulada) de comentarios de usuarios con
   su rating.
"""

from langchain_core.tools import tool

from app.agents.schemas import (
    BuscarComentariosInput,
    BuscarDocumentosInput,
    Comentario,
    ComentariosNoEncontrados,
    ResultadoComentarios,
)
from app.rag.service import get_rag

_COMENTARIOS_SIMULADOS = {
    "checkout": [
        {"texto": "El nuevo checkout es mucho más rápido, me encantó la experiencia.", "rating": 5},
        {"texto": "Tarda demasiado en cargar el paso de pago, es frustrante.", "rating": 2},
        {"texto": "Buena experiencia en general, aunque el botón de confirmar es confuso.", "rating": 3},
        {"texto": "Excelente, el proceso de compra quedó buenísimo, lo recomiendo.", "rating": 5},
        {"texto": "Tuve un error al confirmar la compra, muy mala experiencia.", "rating": 1},
    ],
    "onboarding": [
        {"texto": "El tutorial inicial es claro y rápido, muy bien explicado.", "rating": 5},
        {"texto": "Me perdí en el paso 3, la explicación es confusa.", "rating": 2},
        {"texto": "Buena primera impresión, aunque es un poco largo.", "rating": 3},
    ],
}


@tool(args_schema=BuscarDocumentosInput)
async def buscar_en_documentos(consulta: str, top_k: int = 3) -> dict:
    """
    Busca en la documentación técnica de la empresa (arquitectura,
    seguridad, testing y deployment) con recuperación híbrida y devuelve
    los fragmentos más relevantes con su archivo de origen.

    Usala para cualquier pregunta sobre cómo está construido, probado,
    asegurado o desplegado el sistema. Si los fragmentos devueltos no
    contienen la respuesta, la información NO está en la documentación.
    """
    resultado = await get_rag().buscar(consulta, top_k)
    return resultado.model_dump()


@tool(args_schema=BuscarComentariosInput)
async def buscar_comentarios(tema: str) -> dict:
    """
    Busca comentarios de usuarios sobre una feature del producto, junto
    con el rating (1-5) de cada uno. Usala cuando la solicitud pida
    opiniones o feedback de usuarios.
    """
    tema_normalizado = tema.strip().lower()

    for clave, comentarios in _COMENTARIOS_SIMULADOS.items():
        if clave in tema_normalizado or tema_normalizado in clave:
            return ResultadoComentarios(
                tema=clave, comentarios=[Comentario(**c) for c in comentarios]
            ).model_dump()

    return ComentariosNoEncontrados(
        error=f"No se encontraron comentarios sobre '{tema}'.",
        temas_disponibles=list(_COMENTARIOS_SIMULADOS.keys()),
    ).model_dump()


TOOLS_INVESTIGADOR = [buscar_en_documentos, buscar_comentarios]

PROMPT_INVESTIGADOR = (
    "Sos el Agente de Investigación. Tu única función es RECOLECTAR "
    "información con tus herramientas y devolver un resumen objetivo:\n"
    "- buscar_en_documentos: para preguntas sobre la documentación técnica "
    "(arquitectura, seguridad, testing, deployment). Citá siempre el "
    "archivo de origen de cada dato, entre corchetes, ej: [03_testing.md].\n"
    "- buscar_comentarios: para opiniones de usuarios (texto y rating).\n"
    "Si la solicitud necesita ambas fuentes, usá ambas.\n"
    "REGLA DE ORO: respondé SOLO con lo que devuelvan las herramientas. Si "
    "los fragmentos no contienen la respuesta, decí explícitamente que esa "
    "información no está en la documentación — nunca la inventes.\n"
    "NO analices el sentimiento ni calcules nada: eso es del Agente de Análisis."
)


def crear_agente_investigador(llm):
    """Sub-agente ReAct (LangChain 1.x `create_agent`): LLM + herramientas acotadas. Se invoca con `ainvoke`."""
    from langchain.agents import create_agent

    return create_agent(llm, tools=TOOLS_INVESTIGADOR, system_prompt=PROMPT_INVESTIGADOR)
