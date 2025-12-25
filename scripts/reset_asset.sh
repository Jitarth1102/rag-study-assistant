#!/usr/bin/env bash
set -euo pipefail

ASSET_ID="${1:-}"
if [[ -z "$ASSET_ID" ]]; then
  echo "Usage: $0 <asset_id>" >&2
  exit 1
fi

echo "[reset] asset_id=${ASSET_ID}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/default.yaml"
DEFAULT_DB="$ROOT_DIR/data/db/rag_assistant.db"
DB_PATH="$DEFAULT_DB"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="rag_chunks_e5"

# Read config values if available
if [[ -f "$CONFIG_PATH" ]]; then
  read -r DB_PATH QDRANT_URL QDRANT_COLLECTION <<EOF
$(python - <<'PY' "$CONFIG_PATH" "$DEFAULT_DB" "$QDRANT_URL" "$QDRANT_COLLECTION"
import sys, yaml
config_path, default_db, default_url, default_coll = sys.argv[1:5]
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    db_path = data.get('database', {}).get('sqlite_path') or default_db
    qdrant = data.get('qdrant', {}) or {}
    url = qdrant.get('url', default_url)
    collection = qdrant.get('collection', default_coll)
except Exception:
    db_path, url, collection = default_db, default_url, default_coll
print(db_path)
print(url)
print(collection)
PY
)
EOF
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "[reset] error: SQLite DB not found at $DB_PATH" >&2
  exit 1
fi

echo "[reset] db_path=${DB_PATH}"

run_sql() {
  local sql="$1"
  sqlite3 "$DB_PATH" "$sql" || true
}

echo "[reset] deleting db rows"
run_sql "DELETE FROM notes_chunks WHERE asset_id = '${ASSET_ID}';"
run_sql "DELETE FROM notes WHERE asset_id = '${ASSET_ID}';"
run_sql "DELETE FROM asset_index_status WHERE asset_id = '${ASSET_ID}';"
run_sql "DELETE FROM asset_pages WHERE asset_id = '${ASSET_ID}';"
run_sql "DELETE FROM asset_ocr_pages WHERE asset_id = '${ASSET_ID}';"
run_sql "DELETE FROM chunks WHERE asset_id = '${ASSET_ID}';"
run_sql "UPDATE assets SET status='stored' WHERE asset_id='${ASSET_ID}';"

remove_path() {
  local target="$1"
  if [[ -e "$target" ]]; then
    rm -rf "$target"
  fi
}

echo "[reset] removing generated files"
for dir in "$ROOT_DIR"/data/subjects/*/pages/"${ASSET_ID}"; do
  [[ -e "$dir" ]] && remove_path "$dir"
done
for dir in "$ROOT_DIR"/data/subjects/*/ocr/"${ASSET_ID}"; do
  [[ -e "$dir" ]] && remove_path "$dir"
done
for file in "$ROOT_DIR"/data/subjects/*/processed/chunks/"${ASSET_ID}".jsonl; do
  [[ -e "$file" ]] && remove_path "$file"
done

echo "[reset] removing qdrant points (best effort)"
DELETE_PAYLOAD=$(cat <<JSON
{"filter": {"must": [{"key": "asset_id", "match": {"value": "${ASSET_ID}"}}]}}
JSON
)
if command -v curl >/dev/null 2>&1; then
  if ! curl -s -o /dev/null --fail -X POST "$QDRANT_URL/collections/$QDRANT_COLLECTION/points/delete" \
    -H "Content-Type: application/json" \
    -d "$DELETE_PAYLOAD"; then
    echo "[reset] qdrant deletion skipped (server unreachable?)" >&2
  else
    echo "[reset] qdrant points removed"
  fi
else
  echo "[reset] curl not found; skipping qdrant cleanup" >&2
fi

echo "[reset] done"
