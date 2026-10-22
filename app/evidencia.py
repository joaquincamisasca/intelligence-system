"""
Evidencia: lo que DEVOLVIERON las herramientas, guardado aparte de lo que
"dijeron" los agentes.

Lección de las pruebas con llama3.2 (un modelo de ~3B parámetros): cuando
el LLM resume o sintetiza, a veces reescribe los números (recalcula mal un
promedio, inventa un "7/7 = 1") o descarta un dato que sí tenía. Por eso:

1. Las salidas de las herramientas (validadas con Pydantic) se guardan en
   el estado como EVIDENCIA, sin pasar por el LLM.
2. Al analista se le pasan los datos EXACTOS (textos y ratings), no un
   resumen en prosa del investigador.
3. Las cifras finales (promedio, sentimiento) se recalculan con las mismas
   herramientas sobre los datos recuperados, y se agregan a la respuesta
   como "Datos verificados". Principio: el LLM redacta, el código calcula.
4. La síntesis recibe los fragmentos reales de la documentación, no solo
   el resumen del investigador.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage

_MAX_CARACTERES_FRAGMENTO = 1500


def _contenido_json(mensaje: ToolMessage) -> Any:
    try:
        return json.loads(mensaje.content) if isinstance(mensaje.content, str) else mensaje.content
    except (ValueError, TypeError):
        return None


def extraer(mensajes: list) -> Dict[str, Any]:
    """Evidencia que aportaron las herramientas durante la ejecución de un agente."""
    evidencia: Dict[str, Any] = {"documentos": [], "comentarios": [], "temas": []}
    for m in mensajes:
        if not isinstance(m, ToolMessage):
            continue
        datos = _contenido_json(m)
        if not isinstance(datos, dict):
            continue  # p. ej. un error de validación devuelto como texto

        if m.name == "buscar_en_documentos" and datos.get("fragmentos"):
            evidencia["documentos"].append(
                {"consulta": datos.get("consulta", ""), "fragmentos": datos["fragmentos"]}
            )
        elif m.name == "buscar_comentarios" and datos.get("comentarios"):
            evidencia["temas"].append(datos.get("tema", ""))
            evidencia["comentarios"].extend(datos["comentarios"])
    return evidencia


def fusionar(actual: Dict[str, Any] | None, nueva: Dict[str, Any]) -> Dict[str, Any]:
    """Une la evidencia nueva con la acumulada (sin duplicar comentarios)."""
    resultado = {
        "documentos": list((actual or {}).get("documentos", [])),
        "comentarios": list((actual or {}).get("comentarios", [])),
        "temas": list((actual or {}).get("temas", [])),
    }
    for clave in ("sentimiento", "estadisticas"):
        if clave in (actual or {}):
            resultado[clave] = actual[clave]

    resultado["documentos"].extend(nueva.get("documentos", []))
    vistos = {c["texto"] for c in resultado["comentarios"]}
    for c in nueva.get("comentarios", []):
        if c["texto"] not in vistos:
            resultado["comentarios"].append(c)
            vistos.add(c["texto"])
    for t in nueva.get("temas", []):
        if t not in resultado["temas"]:
            resultado["temas"].append(t)
    for clave in ("sentimiento", "estadisticas"):
        if clave in nueva:
            resultado[clave] = nueva[clave]
    return resultado


_SEÑALES_DOCUMENTALES = ("document", "segun", "manual", "politica", "norma")
_SEÑALES_DE_OPINION = ("comentario", "opin", "usuario", "feedback", "resena")


def es_pregunta_documental(solicitud: str) -> bool:
    """¿La solicitud pide algo de la documentación? (sin tildes, en minúscula)."""
    from app.hitl import _normalizar

    texto = _normalizar(solicitud)
    if any(s in texto for s in _SEÑALES_DOCUMENTALES):
        return True
    return not any(s in texto for s in _SEÑALES_DE_OPINION)


def fuentes_relevantes(evidencia: Dict[str, Any], texto_investigador: str = "", solicitud: str = "") -> List[str]:
    """
    Documentos que de verdad respaldan la respuesta.

    Un documento cuenta si el investigador lo citó, o si la pregunta es sobre
    la documentación y lo encontraron LAS DOS búsquedas (vectorial y léxica:
    coinciden en significado y en términos). Se toma el mejor de cada búsqueda.

    Probado con llama3.2: el investigador a veces consulta la documentación
    aunque la tarea sea solo sobre comentarios; sin este filtro aparecía un
    documento sin relación como "fuente". Y sin quedarse con el mejor, con un
    corpus de 4 archivos se listaban siempre los 4.
    """
    fuentes: List[str] = []
    recuperados: List[str] = []
    documental = es_pregunta_documental(solicitud) if solicitud else True
    for busqueda in evidencia.get("documentos", []):
        frags = busqueda.get("fragmentos", [])
        fuertes = [f for f in frags if f.get("recuperadores", 1) >= 2]
        if documental and fuertes and fuertes[0]["fuente"] not in fuentes:
            fuentes.append(fuertes[0]["fuente"])
        recuperados.extend(f["fuente"] for f in frags)
    for fuente in recuperados:
        if fuente in texto_investigador and fuente not in fuentes:
            fuentes.append(fuente)
    return fuentes


def datos_exactos_para_analista(evidencia: Dict[str, Any]) -> str:
    """Bloque con las listas EXACTAS que el analista debe pasarle a sus herramientas."""
    comentarios = evidencia.get("comentarios", [])
    if not comentarios:
        return ""
    textos = [c["texto"] for c in comentarios]
    ratings = [c["rating"] for c in comentarios]
    return (
        "DATOS EXACTOS (copialos tal cual en tus herramientas, no los resumas):\n"
        f"comentarios = {json.dumps(textos, ensure_ascii=False)}\n"
        f"ratings = {json.dumps(ratings)}"
    )


async def verificar_analisis(evidencia: Dict[str, Any]) -> Dict[str, Any]:
    """Recalcula sentimiento y estadísticas con las MISMAS herramientas, sobre los datos recuperados."""
    from app.agents.analyst_agent import analizar_sentimiento, calcular_estadisticas

    comentarios = evidencia.get("comentarios", [])
    if not comentarios:
        return {}
    return {
        "sentimiento": await analizar_sentimiento.ainvoke({"comentarios": [c["texto"] for c in comentarios]}),
        "estadisticas": await calcular_estadisticas.ainvoke({"numeros": [c["rating"] for c in comentarios]}),
    }


def bloque_para_sintesis(evidencia: Dict[str, Any]) -> str:
    """Evidencia en texto para el prompt de síntesis: fragmentos reales + cifras verificadas."""
    partes: List[str] = []

    for busqueda in evidencia.get("documentos", []):
        frag = busqueda["fragmentos"][0]
        texto = frag["texto"][:_MAX_CARACTERES_FRAGMENTO]
        partes.append(f"[{frag['fuente']}] (fragmento más relevante para '{busqueda['consulta']}'):\n{texto}")

    est = evidencia.get("estadisticas")
    if est:
        partes.append(
            f"Rating promedio VERIFICADO: {est['promedio']} sobre {est['cantidad']} comentarios "
            f"(mínimo {est['minimo']}, máximo {est['maximo']})."
        )
    sen = evidencia.get("sentimiento")
    if sen:
        partes.append(
            f"Sentimiento VERIFICADO: {sen['sentimiento_general']} ({sen['positivos']} positivos, "
            f"{sen['negativos']} negativos, {sen['neutros']} neutros)."
        )
    return "\n\n".join(partes) if partes else "(sin evidencia de herramientas)"


def apendice_verificado(evidencia: Dict[str, Any], fuentes: List[str]) -> str:
    """Sección final de la respuesta armada por código (no por el LLM)."""
    lineas: List[str] = []
    est = evidencia.get("estadisticas")
    if est:
        lineas.append(
            f"- Rating promedio: {est['promedio']} ({est['cantidad']} comentarios; mín. {est['minimo']}, máx. {est['maximo']})"
        )
    sen = evidencia.get("sentimiento")
    if sen:
        lineas.append(
            f"- Sentimiento: {sen['sentimiento_general']} ({sen['positivos']} positivos, "
            f"{sen['negativos']} negativos, {sen['neutros']} neutros)"
        )
    if fuentes:
        lineas.append(f"- Documentación consultada (lo más relevante que encontró el RAG): {', '.join(fuentes)}")
    if not lineas:
        return ""
    return "\n\n---\n**Datos verificados (calculados por las herramientas, no por el LLM):**\n" + "\n".join(lineas)


def resumen(evidencia: Dict[str, Any]) -> Dict[str, Any]:
    """Versión compacta para el resultado de la API."""
    return {
        "temas_de_comentarios": evidencia.get("temas", []),
        "cantidad_comentarios": len(evidencia.get("comentarios", [])),
        "busquedas_en_documentos": [b["consulta"] for b in evidencia.get("documentos", [])],
        "sentimiento": evidencia.get("sentimiento"),
        "estadisticas": evidencia.get("estadisticas"),
    }


# ---------------------------------------------------------------------------
# Guardrail de salida
# ---------------------------------------------------------------------------
# Probado con llama3.2: aun con la instrucción explícita de NO escribir cifras,
# el modelo seguía recalculando promedios en el texto (3.5 en vez de 3.2) y
# agregando una sección "Fuentes" inventada. Las instrucciones no alcanzan con
# un modelo de ~3B: el control tiene que estar en el código.

_PALABRAS_DE_CIFRA = re.compile(r"(promedio|rating|calificaci|puntaje|estrellas)", re.IGNORECASE)
_CONTEO_DE_SENTIMIENTO = re.compile(r"\d.*(positiv|negativ|neutr)|(positiv|negativ|neutr).*\d", re.IGNORECASE)
_TITULO_FUENTES = re.compile(r"^\W*(fuentes?|referencias?)\W*$", re.IGNORECASE)
_ARCHIVO_CITADO = re.compile(r"\[?(\d{2}_[\w-]+\.(?:md|txt))\]?")


def _es_titulo(linea: str) -> bool:
    t = linea.strip()
    if not t or t.startswith(("*", "-", "+")) and not t.startswith("**"):
        return False
    cuerpo = t.strip("*: ")
    return len(t) <= 60 and (t.startswith("**") or t.endswith(":")) and "." not in cuerpo and "," not in cuerpo


def _sacar_titulos_vacios(lineas: List[str]) -> List[str]:
    """Quita los títulos que quedaron sin contenido debajo (después de limpiar)."""
    resultado: List[str] = []
    primera = next((i for i, l in enumerate(lineas) if l.strip()), None)
    for i, linea in enumerate(lineas):
        if _es_titulo(linea) and i != primera:  # el título principal se conserva
            siguiente = next((l for l in lineas[i + 1 :] if l.strip()), None)
            if siguiente is None or _es_titulo(siguiente):
                continue
        resultado.append(linea)
    return resultado


def limpiar_sintesis(texto: str, evidencia: Dict[str, Any], fuentes: List[str]) -> str:
    """
    Quita del texto del LLM lo que el código informa mejor:
    - Si hay cifras verificadas: las líneas con promedios/ratings o conteos de
      sentimiento (se informan en "Datos verificados").
    - Siempre: las secciones "Fuentes" que escribe el LLM y las citas a
      archivos que el RAG no recuperó como relevantes.
    Al final saca los títulos que quedaron vacíos.
    """
    hay_cifras = bool(evidencia.get("estadisticas") or evidencia.get("sentimiento"))
    lineas_limpias: List[str] = []
    modo_fuentes = None  # None | "esperando" (blancos tras el título) | "saltando" (su párrafo)

    for linea in texto.splitlines():
        if modo_fuentes == "esperando":
            if not linea.strip():
                continue
            modo_fuentes = "saltando"
        if modo_fuentes == "saltando":
            if linea.strip():
                continue
            modo_fuentes = None
        if _TITULO_FUENTES.match(linea.strip()):
            modo_fuentes = "esperando"
            continue
        if hay_cifras and (_PALABRAS_DE_CIFRA.search(linea) or _CONTEO_DE_SENTIMIENTO.search(linea)):
            continue
        citados = _ARCHIVO_CITADO.findall(linea)
        if citados and any(c not in fuentes for c in citados):
            linea = _ARCHIVO_CITADO.sub(lambda m: m.group(0) if m.group(1) in fuentes else "", linea).rstrip()
            if not linea.strip(" .:-*"):
                continue
        lineas_limpias.append(linea)

    limpio = "\n".join(_sacar_titulos_vacios(lineas_limpias))
    limpio = re.sub(r"\n{3,}", "\n\n", limpio).strip()
    return limpio or texto.strip()
