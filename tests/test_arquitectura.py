"""
Guardas de arquitectura: revisan el CÓDIGO FUENTE para que los errores
comunes de la consigna no vuelvan a colarse.

- Sincronicidad escondida: nada de `requests`, `time.sleep` ni llamadas
  síncronas `.invoke()` dentro de una función `async def`.
- Hard-coding: ningún secreto ni modelo/índice escrito en el código.
"""

import ast
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).parent.parent
ARCHIVOS_APP = sorted((RAIZ / "app").rglob("*.py"))


def _llamadas_dentro_de_async(arbol: ast.AST):
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.AsyncFunctionDef):
            for hijo in ast.walk(nodo):
                if isinstance(hijo, ast.Call):
                    yield nodo.name, hijo


@pytest.mark.parametrize("archivo", ARCHIVOS_APP, ids=lambda p: str(p.relative_to(RAIZ)))
def test_no_hay_llamadas_bloqueantes_dentro_de_funciones_async(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    problemas = []
    for funcion, llamada in _llamadas_dentro_de_async(arbol):
        f = llamada.func
        nombre = f"{getattr(f.value, 'id', '')}.{f.attr}" if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if nombre == "time.sleep" or nombre.startswith("requests.") or (
            isinstance(f, ast.Attribute) and f.attr == "invoke"
        ):
            problemas.append(f"{funcion}() llama a {nombre or f.attr}")
    assert not problemas, f"Llamadas síncronas dentro de async def: {problemas}"


@pytest.mark.parametrize("archivo", ARCHIVOS_APP, ids=lambda p: str(p.relative_to(RAIZ)))
def test_no_se_importa_requests(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            assert all(a.name.split(".")[0] != "requests" for a in nodo.names)
        if isinstance(nodo, ast.ImportFrom):
            assert (nodo.module or "").split(".")[0] != "requests"


def test_el_repo_no_contiene_api_keys():
    patron = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|pcsk_[A-Za-z0-9_]{20,})")
    for archivo in RAIZ.rglob("*"):
        if archivo.is_file() and archivo.suffix in {".py", ".md", ".yml", ".yaml", ".sh", ".txt", ".example", ".toml", ""}:
            if any(p in archivo.parts for p in ("venv", ".venv", "storage", "__pycache__", ".git", ".pytest_cache")):
                continue
            try:
                texto = archivo.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            assert not patron.search(texto), f"Parece haber una API key en {archivo}"


def test_los_nombres_de_modelo_e_indice_no_estan_escritos_fuera_de_config():
    """Los defaults viven SOLO en app/config.py; el resto del código los lee de `settings`."""
    permitido = {RAIZ / "app" / "config.py"}
    prohibidos = ("llama3.2", "gpt-4o", "claude-3", "text-embedding-3", "intelligence-system\"", "MiniLM")
    for archivo in ARCHIVOS_APP:
        if archivo in permitido:
            continue
        codigo = "\n".join(
            linea for linea in archivo.read_text(encoding="utf-8").splitlines() if not linea.lstrip().startswith("#")
        )
        # se ignoran los docstrings/comentarios: se buscan solo literales de string en el AST
        for nodo in ast.walk(ast.parse(codigo)):
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and len(nodo.value) < 80:
                assert not any(p.strip('"') in nodo.value for p in prohibidos), (
                    f"Literal hard-codeado en {archivo.name}: {nodo.value!r}"
                )
