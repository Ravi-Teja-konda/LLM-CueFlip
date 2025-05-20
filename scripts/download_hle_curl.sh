#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Require HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
  echo "⚠️  Please set your Hugging Face token: export HF_TOKEN=hf_xxx"
  exit 1
fi

OUT_DIR="data/raw/hle"
mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/hle_test.jsonl"
> "$OUT_FILE"

# 2️⃣ Fetch in pages of 500 rows until exhausted
OFFSET=0
PAGE_SIZE=500

while true; do
  echo "📥 Fetching rows offset=$OFFSET..."
  # call the HF datasets-rows API
  RESP=$(curl -s -H "Authorization: Bearer $HF_TOKEN" \
    "https://datasets-server.huggingface.co/rows?dataset=cais%2Fhle&config=default&split=test&offset=${OFFSET}&length=${PAGE_SIZE}")

  # extract the array of rows
  ROWS=$(jq -c '.rows[]' <<<"$RESP")
  COUNT=$(jq '.rows | length' <<<"$RESP")

  # append each row JSON on its own line
  if [ "$COUNT" -gt 0 ]; then
    echo "$ROWS" >> "$OUT_FILE"
  fi

  # stop when fewer than a full page was returned
  if [ "$COUNT" -lt "$PAGE_SIZE" ]; then
    echo "✅ Completed download. Total rows: $((OFFSET + COUNT))"
    break
  fi

  OFFSET=$((OFFSET + PAGE_SIZE))
done
