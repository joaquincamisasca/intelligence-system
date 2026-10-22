"""
Tests del nodo Supervisor (async) con un LLM FALSO.

Cubren el fallo real observado con llama3.2 en Ollama —
`with_structured_output()` devolvió None— y la rúbrica de validación:
no se puede cerrar antes de tiempo ni saltearse al investigador. También
el ruteo dinámico: el analista solo es requerido si la solicitud pide análisis.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app import supervisor
from app.supervisor import DecisionSupervisor, nodo_supervisor, requiere_analisis

SOLICITUD_CON_ANALISIS = "Analizá el feedback del checkout y calculá el rating promedio."
SOLICITUD_DOCUMENTAL = "¿Qué cobertura mínima de tests exige la documentación?"


class _LLMFalso:
    def __init__(self, respuestas_decision):
        self._respuestas = list(respuestas_decision)
        self.intentos_realizados = 0

    def with_structured_output(self, _esquema):
        def _invocar(_mensajes):
            self.intentos_realizados += 1
            valor = self._respuestas.pop(0) if self._respuestas else None
            if isinstance(valor, Exception):
                raise valor
            return valor

        return RunnableLambda(_invocar)

    async def ainvoke(self, _mensajes):
        return AIMessage(content="Síntesis final de prueba.")


def _estado(contribuciones=None, pasos=0, solicitud=SOLICITUD_CON_ANALISIS):
    return {
        "messages": [HumanMessage(content=solicitud)],
        "contribuciones": contribuciones or {},
        "pasos": pasos,
    }


def _usar(monkeypatch, llm):
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        (SOLICITUD_CON_ANALISIS, True),
        ("Calculá el promedio de ratings", True),
        ("¿Cuál es el sentimiento de los usuarios?", True),
        (SOLICITUD_DOCUMENTAL, False),
        ("¿Cómo se gestionan los secretos?", False),
    ],
)
def test_requiere_analisis(texto, esperado):
    assert requiere_analisis(texto) is esperado


async def test_llm_devuelve_none_cae_al_plan_b_y_delega_al_investigador(monkeypatch):
    llm = _LLMFalso([None, None])
    _usar(monkeypatch, llm)

    resultado = await nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"
    assert resultado["instruccion_para_agente"].strip() != ""
    assert llm.intentos_realizados == supervisor.INTENTOS_DECISION


async def test_llm_lanza_excepcion_tambien_cae_al_plan_b(monkeypatch):
    _usar(monkeypatch, _LLMFalso([ValueError("json inválido"), ValueError("json inválido")]))
    assert (await nodo_supervisor(_estado()))["next_agent"] == "investigador"


async def test_reintento_exitoso_no_usa_plan_b(monkeypatch):
    valida = DecisionSupervisor(next_agent="investigador", instruccion="Buscá comentarios.", razon="Faltan datos.")
    _usar(monkeypatch, _LLMFalso([None, valida]))

    resultado = await nodo_supervisor(_estado())

    assert resultado["instruccion_para_agente"] == "Buscá comentarios."


async def test_plan_b_delega_al_analista_con_los_datos_del_investigador(monkeypatch):
    _usar(monkeypatch, _LLMFalso([None, None]))

    resultado = await nodo_supervisor(_estado(contribuciones={"investigador": "5 comentarios, ratings 5,2,3,5,1"}))

    assert resultado["next_agent"] == "analista"
    assert "5 comentarios, ratings 5,2,3,5,1" in resultado["instruccion_para_agente"]


async def test_rubrica_impide_cerrar_antes_de_tiempo(monkeypatch):
    finish = DecisionSupervisor(next_agent="FINISH", razon="Ya está.")
    _usar(monkeypatch, _LLMFalso([finish]))
    assert (await nodo_supervisor(_estado()))["next_agent"] == "investigador"


async def test_rubrica_impide_saltear_al_investigador(monkeypatch):
    salto = DecisionSupervisor(next_agent="analista", instruccion="Analizá.", razon="...")
    _usar(monkeypatch, _LLMFalso([salto]))
    assert (await nodo_supervisor(_estado()))["next_agent"] == "investigador"


async def test_analista_siempre_recibe_los_datos_aunque_el_llm_los_omita(monkeypatch):
    decision = DecisionSupervisor(next_agent="analista", instruccion="Analizá el sentimiento.", razon="...")
    _usar(monkeypatch, _LLMFalso([decision]))

    resultado = await nodo_supervisor(_estado(contribuciones={"investigador": "DATOS-DEL-INVESTIGADOR"}))

    assert "DATOS-DEL-INVESTIGADOR" in resultado["instruccion_para_agente"]


async def test_cierra_con_sintesis_cuando_ambos_contribuyeron(monkeypatch):
    finish = DecisionSupervisor(next_agent="FINISH", razon="Completo.")
    _usar(monkeypatch, _LLMFalso([finish]))

    resultado = await nodo_supervisor(_estado(contribuciones={"investigador": "datos", "analista": "resultados"}))

    assert resultado["next_agent"] == "FINISH"
    assert resultado["messages"][-1].content == "Síntesis final de prueba."
    assert resultado["messages"][-1].name == "supervisor"


async def test_pregunta_documental_cierra_tras_el_investigador_sin_analista(monkeypatch):
    """Ruteo dinámico: sin pedido de análisis, el investigador alcanza."""
    _usar(monkeypatch, _LLMFalso([None, None]))

    resultado = await nodo_supervisor(
        _estado(contribuciones={"investigador": "80% [03_testing.md]"}, solicitud=SOLICITUD_DOCUMENTAL)
    )

    assert resultado["next_agent"] == "FINISH"


async def test_no_re_delega_si_ya_contribuyeron_todos_los_requeridos(monkeypatch):
    """Criterio de Suficiencia en código: aunque el LLM pida otra vuelta, se cierra."""
    otra_vuelta = DecisionSupervisor(next_agent="investigador", instruccion="Buscá más.", razon="Perfeccionismo.")
    _usar(monkeypatch, _LLMFalso([otra_vuelta]))

    resultado = await nodo_supervisor(
        _estado(contribuciones={"investigador": "datos"}, solicitud=SOLICITUD_DOCUMENTAL)
    )

    assert resultado["next_agent"] == "FINISH"


async def test_corte_por_max_pasos_no_llama_al_llm(monkeypatch):
    def _no_debe_llamarse():
        raise AssertionError("No debería llamarse al LLM después de MAX_PASOS")

    monkeypatch.setattr(supervisor, "get_llm", _no_debe_llamarse)

    resultado = await nodo_supervisor(_estado(pasos=supervisor.settings.max_pasos_supervisor))

    assert resultado["next_agent"] == "FINISH"
    assert "máximo" in resultado["messages"][-1].content


async def test_la_delegacion_queda_registrada_en_la_traza(monkeypatch):
    _usar(monkeypatch, _LLMFalso([None, None]))

    resultado = await nodo_supervisor(_estado())

    assert resultado["messages"][-1].name == "supervisor"
    assert "investigador" in resultado["messages"][-1].content
