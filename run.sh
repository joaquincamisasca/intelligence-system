#!/usr/bin/env bash
# Levanta todo el sistema con UNA instrucción:   ./run.sh
# (en Windows usá directamente:  docker compose up --build)
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Falta Docker. Instalalo desde https://www.docker.com/products/docker-desktop/" >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example (revisalo si querés usar OpenAI/Anthropic/Pinecone)."
fi

echo "Levantando API + Redis Stack + Phoenix ..."
echo "  API:     http://localhost:8000/docs"
echo "  Phoenix: http://localhost:6006"
exec docker compose up --build "$@"
