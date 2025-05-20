#!/usr/bin/env bash
set -euo pipefail

# Define your input and output paths
PARQUET_FILE="hle/data/test-00000-of-00001.parquet"
JSONL_OUT="data/raw/hle/hle_test.jsonl"



python3 <<PY
import pyarrow.parquet as pq
import json
from pathlib import Path

# Paths (expanded by Bash)
parquet_path = "${PARQUET_FILE}"
jsonl_path   = "${JSONL_OUT}"

# 1) Load the Parquet table
table = pq.read_table(parquet_path)

# 2) Convert to pandas so we can easily drop unwanted columns
df = table.to_pandas()

# 3) Keep only the MCQ text fields (drop binary image columns)
#    Adjust these names to whatever your dataset uses
keep = ["id", "question", "choices", "answer"]
for col in list(df.columns):
    if col not in keep:
        df.drop(columns=col, inplace=True)

# 4) Write JSONL, one record per line
out_path = Path(jsonl_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for record in df.to_dict(orient="records"):
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")

print(f"✅ Converted {len(df)} rows → {out_path}")
PY
