"""
Ingesta manual de los documentos a la base vectorial.

No hace falta correrla a mano: la API indexa sola al arrancar si la base
está vacía. Sirve para forzar un reindexado (por ejemplo, después de
agregar documentos a data/documentos/) o para preparar la base antes.

Uso (desde la raíz del proyecto):
    python -m scripts.ingest
    python -m scripts.ingest --forzar
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.service import get_rag  # noqa: E402


async def main(forzar: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rag = get_rag()
    await rag.inicializar(forzar_reindexado=forzar)
    print(f"Base vectorial lista ({rag.store_nombre}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos al RAG")
    parser.add_argument("--forzar", action="store_true", help="reindexar aunque la base ya tenga datos")
    args = parser.parse_args()
    asyncio.run(main(args.forzar))
