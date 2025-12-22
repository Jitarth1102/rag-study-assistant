#!/usr/bin/env bash
set -euo pipefail

VOLUME_NAME="qdrant_rag_data"
CONTAINER_NAME="qdrant_rag"
PORT=${QDRANT_PORT:-6333}

echo "Starting Qdrant on port ${PORT} with volume ${VOLUME_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:6333" \
  -v "${VOLUME_NAME}:/qdrant/storage" \
  qdrant/qdrant
