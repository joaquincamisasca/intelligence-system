"""
Contratos Pydantic de las HERRAMIENTAS de los agentes.

Cada herramienta valida su ENTRADA con un `args_schema` (si el LLM manda
argumentos inválidos, LangChain devuelve el error de validación al agente
como mensaje y el agente puede corregirse: es el ciclo de retorno) y su
SALIDA con un modelo (lo que vuelve al agente siempre tiene forma conocida).
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Agente investigador
# ---------------------------------------------------------------------------

class BuscarDocumentosInput(BaseModel):
    consulta: str = Field(
        min_length=3,
        max_length=500,
        description="Pregunta o tema a buscar en la documentación técnica, en lenguaje natural.",
    )
    top_k: int = Field(default=3, ge=1, le=8, description="Cantidad de fragmentos a recuperar (1 a 8).")

    model_config = ConfigDict(str_strip_whitespace=True)


class BuscarComentariosInput(BaseModel):
    tema: str = Field(
        min_length=2,
        max_length=100,
        description='Feature sobre la que se buscan comentarios de usuarios, ej: "checkout".',
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class Comentario(BaseModel):
    texto: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)


class ResultadoComentarios(BaseModel):
    tema: str
    comentarios: List[Comentario]


class ComentariosNoEncontrados(BaseModel):
    error: str
    temas_disponibles: List[str]


# ---------------------------------------------------------------------------
# Agente analista
# ---------------------------------------------------------------------------

class AnalizarSentimientoInput(BaseModel):
    comentarios: List[str] = Field(
        min_length=1, description="Textos a clasificar (ej: comentarios de usuarios)."
    )


class ResultadoSentimiento(BaseModel):
    positivos: int = Field(ge=0)
    negativos: int = Field(ge=0)
    neutros: int = Field(ge=0)
    sentimiento_general: Literal["positivo", "negativo", "mixto"]


class CalcularEstadisticasInput(BaseModel):
    numeros: List[float] = Field(min_length=1, description="Valores numéricos (ej: ratings).")


class ResultadoEstadisticas(BaseModel):
    promedio: float
    minimo: float
    maximo: float
    cantidad: int = Field(ge=1)
