#!/usr/bin/env bash
# Download MMLU test split into data/raw/mmlu using 🤗 Datasets
set -euo pipefail
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
print("📥 Downloading cais/mmlu:test → data/raw/mmlu …")
ds = load_dataset("cais/mmlu", "all", split="test")
Path("data/raw/mmlu").mkdir(parents=True, exist_ok=True)
out = Path("data/raw/mmlu/mmlu_test.jsonl")
ds.to_json(out, lines=True)
print("✅ Saved", len(ds), "items to", out)
PY
