"""
Esquema de Estado compartido del orquestador (heredado del Módulo 6).

Campos (además de `messages`, que viene de MessagesState):

- next_agent: a quién le toca actuar ahora, según decide el Supervisor.
- instruccion_para_agente: instrucción puntual para el especialista
  elegido (evita la "Contaminación de Contexto": el especialista no ve
  todo el historial, solo esto).
- contribuciones: qué aportó cada agente.
- pasos: contador de vueltas del Supervisor (salvaguarda contra el
  "Supervisor Infinito").
- fuentes: archivos de la documentación citados por el RAG (referencias
  de la respuesta; vacío si la tarea no consultó la documentación).
- evidencia: lo que DEVOLVIERON las herramientas (fragmentos del RAG,
  comentarios, cifras verificadas), guardado aparte de lo que dijeron los
  agentes. Ver app/evidencia.py.
- publicacion: resultado de la acción crítica (publicar el reporte), que
  solo se ejecuta después de la aprobación humana. Queda en None si la
  tarea no era crítica.
"""

from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import MessagesState

NombreAgente = Literal["investigador", "analista", "FINISH"]


class OrquestadorState(MessagesState):
    """Estado compartido entre el Supervisor, los especialistas y el nodo de aprobación humana."""

    next_agent: NombreAgente
    instruccion_para_agente: str
    contribuciones: Dict[str, str]
    pasos: int
    fuentes: List[str]
    evidencia: Dict[str, Any]
    publicacion: Optional[dict]
