"""
Configuración compartida de los tests.

Los tests corren SIN Redis Stack, SIN Phoenix, SIN Ollama y SIN internet:
- Phoenix desactivado.
- Embeddings "hash" (determinísticos, sin descargar modelos) y una base
  ChromaDB REAL en una carpeta temporal.
- Un chat model FALSO y guionado que hace de Supervisor, Investigador y
  Analista. Además simula el fallo real de llama3.2 (la decisión
  estructurada del Supervisor vuelve None), así que todos los tests
  ejercitan también el plan B por reglas.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="intelligence-tests-")
os.environ["PHOENIX_ENABLED"] = "false"
os.environ["EMBEDDINGS_PROVIDER"] = "hash"
os.environ["VECTOR_STORE"] = "chroma"
os.environ["CHROMA_PATH"] = os.path.join(_TMP, "chroma")
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "checkpoints.sqlite")
os.environ["GENERATION_PROVIDER"] = "ollama"

import json  # noqa: E402
from typing import Any, List, Optional  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

COMENTARIOS = [
    "El nuevo checkout es mucho más rápido, me encantó la experiencia.",
    "Tarda demasiado en cargar el paso de pago, es frustrante.",
    "Tuve un error al confirmar la compra, muy mala experiencia.",
]
RATINGS = [5, 2, 1]


class ChatGuionado(BaseChatModel):
    """Responde según el rol que indica el system prompt (no llama a ningún LLM real)."""

    @property
    def _llm_type(self) -> str:
        return "chat-guionado"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, _esquema, **kwargs):
        # Simula el fallo real de llama3.2: la salida estructurada vuelve None.
        return RunnableLambda(lambda _mensajes: None)

    def _investigador(self, messages: List[BaseMessage]) -> AIMessage:
        instruccion = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "").lower()
        a_llamar = []
        if "documentaci" in instruccion:
            a_llamar.append(("buscar_en_documentos", {"consulta": "cobertura mínima de tests pull request"}))
        if "checkout" in instruccion or "comentario" in instruccion:
            a_llamar.append(("buscar_comentarios", {"tema": "checkout"}))
        if not a_llamar:
            a_llamar.append(("buscar_comentarios", {"tema": "checkout"}))

        resultados = [m for m in messages if isinstance(m, ToolMessage)]
        if len(resultados) < len(a_llamar):
            nombre, args = a_llamar[len(resultados)]
            return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": f"c{len(resultados)}"}])

        resumen = " | ".join(str(r.content)[:120] for r in resultados)
        return AIMessage(content=f"Datos recolectados. Textos: {COMENTARIOS}. Ratings: {RATINGS}. Fuentes: {resumen}")

    def _analista(self, messages: List[BaseMessage]) -> AIMessage:
        resultados = [m for m in messages if isinstance(m, ToolMessage)]
        if len(resultados) == 0:
            return AIMessage(
                content="",
                tool_calls=[{"name": "analizar_sentimiento", "args": {"comentarios": COMENTARIOS}, "id": "a1"}],
            )
        if len(resultados) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "calcular_estadisticas", "args": {"numeros": RATINGS}, "id": "a2"}],
            )
        return AIMessage(content="Sentimiento mixto (1 positivo, 2 negativos). Rating promedio: 2.67.")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = str(messages[0].content) if messages else ""

        if "Agente de Investigación" in system:
            respuesta = self._investigador(messages)
        elif "Agente de Análisis" in system:
            respuesta = self._analista(messages)
        else:
            respuesta = AIMessage(content="Síntesis: el checkout tiene sentimiento mixto y rating 2.67.")

        return ChatResult(generations=[ChatGeneration(message=respuesta)])


@pytest.fixture
def modelo_falso(monkeypatch):
    """Reemplaza el LLM real por el modelo guionado en todo el orquestador."""
    from app import supervisor

    modelo = ChatGuionado()
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: modelo)
    return modelo


def contenido_json(mensaje) -> dict:
    return json.loads(mensaje.content)
