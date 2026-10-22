"""
Contratos de datos (Pydantic v2) del módulo RAG.

Todo lo que cruza una frontera del RAG (lo que devuelve una búsqueda,
lo que recibe la herramienta del agente) pasa por estos modelos.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Fragmento(BaseModel):
    """Un fragmento (chunk) recuperado de la base documental."""

    fuente: str = Field(description="Archivo de origen, ej: 02_seguridad.md")
    categoria: str = Field(description="Categoría derivada del nombre del archivo, ej: seguridad")
    chunk_index: int = Field(ge=0, description="Posición del fragmento dentro de su archivo")
    texto: str = Field(min_length=1, description="Contenido del fragmento")
    score: float = Field(description="Puntaje de relevancia (mayor = más relevante)")
    recuperadores: int = Field(
        1, ge=1, le=2,
        description="Cuántas búsquedas lo encontraron (2 = la vectorial Y la léxica coinciden: señal fuerte de relevancia)",
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class ResultadoRAG(BaseModel):
    """Salida de una búsqueda híbrida: lo que la herramienta le devuelve al agente."""

    consulta: str
    fragmentos: List[Fragmento]
    total: int = Field(ge=0)
    fuentes: List[str] = Field(description="Archivos distintos citados, en orden de relevancia")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "consulta": "¿Qué porcentaje mínimo de cobertura se exige?",
                "fragmentos": [
                    {
                        "fuente": "03_testing.md",
                        "categoria": "testing",
                        "chunk_index": 0,
                        "texto": "Se exige una cobertura mínima del 80%...",
                        "score": 0.0327,
                    }
                ],
                "total": 1,
                "fuentes": ["03_testing.md"],
            }
        }
    )
