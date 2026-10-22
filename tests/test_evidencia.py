"""
Tests de la capa de EVIDENCIA (lección de las pruebas reales con llama3.2):
las cifras salen de las herramientas, no del LLM; el analista recibe datos
exactos; la síntesis recibe los fragmentos reales; y las fuentes son solo
las relevantes.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app import evidencia, supervisor
from app.config import settings
from app.supervisor import nodo_supervisor

COMENTARIOS = [{"texto": "Me encantó, excelente.", "rating": 5}, {"texto": "Tarda demasiado, frustrante.", "rating": 2}]


def _tool(nombre, datos):
    return ToolMessage(content=json.dumps(datos, ensure_ascii=False), name=nombre, tool_call_id="x")


def _rag(consulta, *fuentes):
    return {
        "consulta": consulta,
        "fragmentos": [
            # el primero lo encontraron las dos búsquedas (relevante); el resto, una sola
            {"fuente": f, "categoria": "x", "chunk_index": 0, "texto": f"contenido de {f}", "score": 0.1,
             "recuperadores": 2 if i == 0 else 1}
            for i, f in enumerate(fuentes)
        ],
        "total": len(fuentes),
        "fuentes": list(fuentes),
    }


def test_extrae_lo_que_devolvieron_las_herramientas():
    mensajes = [
        HumanMessage(content="instrucción"),
        _tool("buscar_en_documentos", _rag("cobertura", "03_testing.md", "01_arquitectura.md")),
        _tool("buscar_comentarios", {"tema": "checkout", "comentarios": COMENTARIOS}),
        ToolMessage(content="Error: 1 validation error", name="calcular_estadisticas", tool_call_id="y"),
        AIMessage(content="resumen del agente"),
    ]
    ev = evidencia.extraer(mensajes)
    assert ev["temas"] == ["checkout"]
    assert len(ev["comentarios"]) == 2
    assert ev["documentos"][0]["fragmentos"][0]["fuente"] == "03_testing.md"


def test_fusionar_no_duplica_comentarios():
    ev = evidencia.fusionar({"comentarios": COMENTARIOS, "temas": ["checkout"]}, {"comentarios": COMENTARIOS, "temas": ["checkout"]})
    assert len(ev["comentarios"]) == 2
    assert ev["temas"] == ["checkout"]


def test_fuentes_relevantes_es_el_mejor_de_cada_busqueda_mas_lo_citado():
    ev = {"documentos": [_rag("q1", "03_testing.md", "01_arquitectura.md", "02_seguridad.md")]}
    assert evidencia.fuentes_relevantes(ev, "") == ["03_testing.md"]
    assert evidencia.fuentes_relevantes(ev, "ver [02_seguridad.md]") == ["03_testing.md", "02_seguridad.md"]


def test_un_documento_que_solo_encontro_una_busqueda_no_cuenta_como_fuente():
    """Caso real con llama3.2: consultó la documentación en una tarea solo de comentarios."""
    ev = {"documentos": [{"consulta": "onboarding", "fragmentos": [
        {"fuente": "02_seguridad.md", "categoria": "x", "chunk_index": 0, "texto": "t", "score": 0.008, "recuperadores": 1}
    ]}]}
    assert evidencia.fuentes_relevantes(ev, "") == []


async def test_verificacion_recalcula_con_las_herramientas():
    verificado = await evidencia.verificar_analisis({"comentarios": COMENTARIOS})
    assert verificado["estadisticas"]["promedio"] == 3.5
    assert verificado["sentimiento"]["positivos"] == 1
    assert verificado["sentimiento"]["negativos"] == 1


def test_el_apendice_lo_arma_el_codigo():
    ev = {"estadisticas": {"promedio": 3.2, "cantidad": 5, "minimo": 1, "maximo": 5},
          "sentimiento": {"sentimiento_general": "mixto", "positivos": 2, "negativos": 2, "neutros": 1}}
    texto = evidencia.apendice_verificado(ev, ["03_testing.md"])
    assert "Rating promedio: 3.2" in texto
    assert "mixto" in texto and "03_testing.md" in texto
    assert evidencia.apendice_verificado({}, []) == ""


class _LLMQueCaptura:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, _esquema):
        return RunnableLambda(lambda _m: None)  # fuerza el plan B

    async def ainvoke(self, mensajes):
        self.prompts.append(mensajes)
        return AIMessage(content="Respuesta redactada por el LLM.")


async def test_el_analista_recibe_los_ratings_exactos(monkeypatch):
    monkeypatch.setattr(supervisor, "get_llm", lambda: _LLMQueCaptura())
    estado = {
        "messages": [HumanMessage(content="Analizá el checkout y calculá el promedio")],
        "contribuciones": {"investigador": "Encontré comentarios (resumen en prosa)."},
        "pasos": 0,
        "evidencia": {"comentarios": COMENTARIOS},
    }
    resultado = await nodo_supervisor(estado)
    assert resultado["next_agent"] == "analista"
    assert "ratings = [5, 2]" in resultado["instruccion_para_agente"]


async def test_la_sintesis_recibe_los_fragmentos_y_agrega_datos_verificados(monkeypatch):
    llm = _LLMQueCaptura()
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)
    estado = {
        "messages": [HumanMessage(content="¿Qué cobertura de tests se exige?")],
        "contribuciones": {"investigador": "No estoy seguro."},
        "pasos": 1,
        "fuentes": ["03_testing.md"],
        "evidencia": {"documentos": [_rag("cobertura", "03_testing.md")]},
    }
    resultado = await nodo_supervisor(estado)

    prompt_sistema = llm.prompts[-1][0]["content"]
    assert "contenido de 03_testing.md" in prompt_sistema  # el fragmento REAL llega a la síntesis
    final = resultado["messages"][-1].content
    assert final.startswith("Respuesta redactada por el LLM.")
    assert "Documentación consultada" in final and "03_testing.md" in final


def test_ollama_usa_una_ventana_de_contexto_amplia():
    from app.llm import get_llm

    assert get_llm("ollama").num_ctx == settings.ollama_num_ctx >= 8192


async def test_con_cifras_verificadas_la_sintesis_no_ve_la_prosa_del_analista(monkeypatch):
    """La prosa del analista puede traer números mal copiados: si hay cifras verificadas, no llega a la síntesis."""
    llm = _LLMQueCaptura()
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)
    estado = {
        "messages": [HumanMessage(content="Analizá el checkout y calculá el promedio")],
        "contribuciones": {"investigador": "5 comentarios", "analista": "PROMEDIO-INVENTADO 7/7 = 1"},
        "pasos": 2,
        "evidencia": {
            "comentarios": COMENTARIOS,
            "estadisticas": {"promedio": 3.5, "cantidad": 2, "minimo": 2, "maximo": 5},
        },
    }
    resultado = await nodo_supervisor(estado)

    prompt_sistema = llm.prompts[-1][0]["content"]
    assert "PROMEDIO-INVENTADO" not in prompt_sistema
    assert "Rating promedio VERIFICADO: 3.5" in prompt_sistema
    assert "Rating promedio: 3.5" in resultado["messages"][-1].content


# --- Guardrail de salida: casos REALES de la corrida con llama3.2 ---

EV_CON_CIFRAS = {"estadisticas": {"promedio": 3.2, "cantidad": 5, "minimo": 1, "maximo": 5}}


def test_guardrail_saca_los_promedios_que_recalcula_el_llm():
    texto = (
        "En cuanto al checkout, algunos usuarios destacan la rapidez.\n\n"
        "Rating promedio: El rating promedio de los comentarios es de 3.5 sobre 5.\n"
        "El sentimiento general es mixto, con 3 comentarios positivos y 2 negativos."
    )
    limpio = evidencia.limpiar_sintesis(texto, EV_CON_CIFRAS, [])
    assert "3.5" not in limpio
    assert "3 comentarios positivos" not in limpio
    assert "algunos usuarios destacan la rapidez" in limpio


def test_guardrail_saca_la_seccion_fuentes_inventada():
    texto = "Reporte del checkout: los usuarios valoran la rapidez.\n\n**Fuentes:**\n\n[03_testing.md]\n"
    limpio = evidencia.limpiar_sintesis(texto, EV_CON_CIFRAS, [])
    assert "Fuentes" not in limpio and "03_testing" not in limpio


def test_guardrail_conserva_las_citas_validas_y_los_datos_de_la_documentacion():
    texto = "Se exige una cobertura mínima del 80% [03_testing.md]. No se cita [02_seguridad.md] acá."
    limpio = evidencia.limpiar_sintesis(texto, {}, ["03_testing.md"])
    assert "80%" in limpio  # sin cifras verificadas, los números de la documentación quedan
    assert "[03_testing.md]" in limpio
    assert "02_seguridad" not in limpio


def test_en_una_tarea_de_comentarios_un_documento_no_citado_no_es_fuente():
    ev = {"documentos": [_rag("checkout", "03_testing.md")]}  # lo encontraron las dos búsquedas
    assert evidencia.fuentes_relevantes(ev, "", "Buscá los comentarios sobre el checkout") == []
    assert evidencia.fuentes_relevantes(ev, "", "Según la documentación, ¿qué cobertura se exige?") == ["03_testing.md"]
    assert evidencia.fuentes_relevantes(ev, "", "¿Cómo se gestionan los secretos?") == ["03_testing.md"]


def test_guardrail_saca_titulos_vacios_pero_no_el_titulo_principal_ni_frases():
    texto = (
        "**Reporte de Checkout**\n\n**Rating promedio:** 3.5\n\n**Comentarios:**\n\n* Destacan la rapidez.\n\n"
        "**Análisis Sentimental**\n\nEl promedio es 3,2/5.\n\n**Recomendaciones**\n\n"
        "Basado en los comentarios, se recomienda:\n\n* Mejorar el paso de pago."
    )
    limpio = evidencia.limpiar_sintesis(texto, EV_CON_CIFRAS, [])
    assert limpio.startswith("**Reporte de Checkout**")
    assert "Análisis Sentimental" not in limpio  # quedó vacío
    assert "**Recomendaciones**" in limpio and "se recomienda:" in limpio
    assert "3.5" not in limpio and "3,2" not in limpio
