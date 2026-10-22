# Imagen de la API (Python 3.12). Se construye con `docker compose up --build`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/cache/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/cache/fastembed

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Se descargan UNA vez, al construir la imagen, para que el primer arranque
# no dependa de internet ni demore: el vocabulario de tiktoken (chunking) y
# el modelo de embeddings multilingüe (~220 MB).
RUN mkdir -p /opt/cache/tiktoken /opt/cache/fastembed \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').embed(['calentamiento']))"

COPY . .

EXPOSE 8000

# La base vectorial (Chroma) y el respaldo SQLite viven en /app/storage (volumen).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
