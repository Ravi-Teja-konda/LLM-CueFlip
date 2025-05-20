#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Ensure you have a Hugging Face token in HF_API_TOKEN
if [ -z "${HF_API_TOKEN:-}" ]; then
  echo "⚠️  Please set your Hugging Face token in the HF_API_TOKEN env var."
  echo "    You can get one at https://huggingface.co/settings/tokens"
  exit 1
fi

# 2️⃣ Download HLE test split with authentication
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import os

token ="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"
print("📥 Downloading gated dataset cais/hle:test → data/raw/hle …")
ds = load_dataset("cais/hle", split="test", use_auth_token=token)
out_dir = Path("data/raw/hle")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "hle_test.jsonl"
ds.to_json(out_file, lines=True)
print(f"✅ Saved {len(ds)} items to {out_file}")
PY