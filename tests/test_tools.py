"""
Tests de las herramientas de los agentes (async) y de su validación Pydantic:
entradas inválidas son rechazadas ANTES de ejecutar la herramienta, y las
salidas siempre tienen la forma del modelo.
"""

import pytest
from pydantic import ValidationError

from app.agents.analyst_agent import analizar_sentimiento, calcular_estadisticas
from app.agents.research_agent import buscar_comentarios, buscar_en_documentos
from app.agents.schemas import (
    BuscarDocumentosInput,
    ResultadoComentarios,
    ResultadoEstadisticas,
    ResultadoSentimiento,
)
from app.rag.schemas import ResultadoRAG


async def test_buscar_comentarios_encuentra_tema_existente():
    resultado = await buscar_comentarios.ainvoke({"tema": "checkout"})
    ResultadoComentarios.model_validate(resultado)  # la salida cumple el contrato
    assert len(resultado["comentarios"]) == 5


async def test_buscar_comentarios_tema_inexistente_devuelve_sugerencias():
    resultado = await buscar_comentarios.ainvoke({"tema": "facturación"})
    assert "error" in resultado
    assert "checkout" in resultado["temas_disponibles"]


async def test_analizar_sentimiento_clasifica_correctamente():
    resultado = await analizar_sentimiento.ainvoke(
        {"comentarios": ["Me encantó, excelente experiencia.", "Tarda demasiado, muy frustrante.", "Todo bien."]}
    )
    ResultadoSentimiento.model_validate(resultado)
    assert resultado["positivos"] == 2
    assert resultado["negativos"] == 1


async def test_calcular_estadisticas_devuelve_valores_correctos():
    resultado = await calcular_estadisticas.ainvoke({"numeros": [5, 2, 3, 5, 1]})
    ResultadoEstadisticas.model_validate(resultado)
    assert resultado["promedio"] == 3.2
    assert (resultado["minimo"], resultado["maximo"], resultado["cantidad"]) == (1, 5, 5)


async def test_buscar_en_documentos_devuelve_un_resultado_rag_valido():
    resultado = await buscar_en_documentos.ainvoke({"consulta": "¿Qué cobertura mínima de tests se exige?"})
    modelo = ResultadoRAG.model_validate(resultado)
    assert modelo.total == len(modelo.fragmentos) > 0
    assert modelo.fuentes[0] == "03_testing.md"


# --- Validación de ENTRADAS: el error ocurre antes de ejecutar la herramienta ---

async def test_lista_vacia_es_rechazada_por_el_esquema():
    with pytest.raises(ValidationError):
        await analizar_sentimiento.ainvoke({"comentarios": []})
    with pytest.raises(ValidationError):
        await calcular_estadisticas.ainvoke({"numeros": []})


async def test_numeros_que_no_son_numeros_son_rechazados():
    with pytest.raises(ValidationError):
        await calcular_estadisticas.ainvoke({"numeros": ["abc"]})


async def test_consulta_demasiado_corta_es_rechazada():
    with pytest.raises(ValidationError):
        await buscar_en_documentos.ainvoke({"consulta": "ab"})


async def test_top_k_fuera_de_rango_es_rechazado():
    with pytest.raises(ValidationError):
        await buscar_en_documentos.ainvoke({"consulta": "cobertura de tests", "top_k": 99})


def test_los_esquemas_de_entrada_limpian_espacios():
    assert BuscarDocumentosInput(consulta="   cobertura de tests   ").consulta == "cobertura de tests"
