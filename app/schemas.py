"""
Contratos de la API (Pydantic v2): TODA entrada y TODA salida de los
endpoints pasa por uno de estos modelos. FastAPI los usa para validar,
para serializar y para generar la documentación interactiva (/docs).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EstadoTrabajo = Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "DONE", "FAILED"]


class TareaEntrada(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Solicitud en lenguaje natural.")

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": (
                    "Buscá en la documentación qué cobertura mínima de tests se exige y "
                    "analizá el sentimiento y el rating promedio de los comentarios sobre el checkout."
                )
            }
        },
    )


class TareaCreada(BaseModel):
    job_id: str
    status: EstadoTrabajo
    requiere_aprobacion: bool


class Aprobacion(BaseModel):
    aprobado: bool = Field(True, description="True aprueba la acción crítica; False la rechaza.")
    comentario: str = Field("", max_length=500, description="Comentario opcional del revisor humano.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"aprobado": True, "comentario": "Revisado, OK para publicar."}}
    )


class AprobacionAceptada(BaseModel):
    job_id: str
    status: EstadoTrabajo
    aprobado: bool


class PasoTraza(BaseModel):
    agente: str
    contenido: str


class ResultadoTarea(BaseModel):
    respuesta: str
    contribuciones: Dict[str, str] = Field(default_factory=dict)
    fuentes: List[str] = Field(default_factory=list, description="Documentos que respaldan la respuesta (el más relevante de cada búsqueda RAG).")
    datos_verificados: Optional[dict] = Field(None, description="Cifras calculadas por las herramientas sobre los datos recuperados.")
    pasos_supervisor: int = 0
    publicacion: Optional[dict] = None
    traza: List[PasoTraza] = Field(default_factory=list)


class Trabajo(BaseModel):
    job_id: str
    status: EstadoTrabajo
    query: str
    requiere_aprobacion: bool
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duracion_s: Optional[float] = None
    result: Optional[ResultadoTarea] = None
    interrupt: Optional[dict] = Field(None, description="Qué se le pide aprobar al humano (WAITING_APPROVAL).")
    error: Optional[str] = None


class CheckpointEstado(BaseModel):
    """Lo que el checkpointer (Redis/SQLite) tiene guardado de una ejecución."""

    job_id: str
    guardado: bool
    checkpointer: str
    siguiente_nodo: List[str] = Field(default_factory=list, description="Vacío si la ejecución terminó.")
    pasos_supervisor: int = 0
    mensajes: int = 0
    contribuciones: List[str] = Field(default_factory=list, description="Agentes que ya contribuyeron.")


class BusquedaRAG(BaseModel):
    consulta: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(3, ge=1, le=8)

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"consulta": "¿Cómo se gestionan los secretos?", "top_k": 3}},
    )


class Salud(BaseModel):
    status: Literal["ok"]
    redis: str
    checkpointer: str
    vector_store: str
    llm: str
