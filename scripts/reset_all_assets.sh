#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/default.yaml"
DEFAULT_DB="$ROOT_DIR/data/db/rag_assistant.db"
DB_PATH="$DEFAULT_DB"
RESET_SCRIPT="$ROOT_DIR/scripts/reset_asset.sh"

if [[ ! -x "$RESET_SCRIPT" ]]; then
  echo "[reset-all] reset_asset.sh not executable or missing" >&2
  exit 1
fi

# Resolve DB path from config if available
if [[ -f "$CONFIG_PATH" ]]; then
  DB_PATH=$(python - <<'PY' "$CONFIG_PATH" "$DEFAULT_DB"
import sys, yaml
config_path, default_db = sys.argv[1:3]
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    db_path = data.get('database', {}).get('sqlite_path') or default_db
    print(db_path)
except Exception:
    print(default_db)
PY
  )
fi

if [[ ! -f "$DB_PATH" ]]; then
  echo "[reset-all] SQLite DB not found at $DB_PATH" >&2
  exit 1
fi

echo "[reset-all] db_path=${DB_PATH}"
ASSETS=$(sqlite3 "$DB_PATH" "SELECT asset_id FROM assets;")
if [[ -z "$ASSETS" ]]; then
  echo "[reset-all] no assets found"
  exit 0
fi
COUNT=0
for aid in $ASSETS; do
  echo "[reset-all] resetting $aid"
  "$RESET_SCRIPT" "$aid"
  COUNT=$((COUNT+1))
done

echo "[reset-all] reset $COUNT assets"
