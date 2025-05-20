#!/usr/bin/env python3
"""Create a random N‑item TSV subset (id\tquestion\tchoices\tcorrect_idx) with only 2 choices per question."""
import argparse, random, json, pandas as pd
from pathlib import Path
from datasets import load_dataset
import yaml

# -------------------- args & config
d = argparse.ArgumentParser()
d.add_argument("--n", type=int, default=200)
d.add_argument("--seed", type=int, default=42)
args = d.parse_args()

paths = yaml.safe_load(open("configs/paths.yaml"))
subset_path = Path(paths["subset_tsv"])
subset_path.parent.mkdir(parents=True, exist_ok=True)

print(f"📖 Loading MMLU test split ({args.n} items) …")
full = load_dataset("cais/mmlu", "all", split="test")
print("📊", len(full), "total items available → sampling", args.n)

# -------------------- sampling setup
rng = random.Random(args.seed)
indices = rng.sample(range(len(full)), args.n)
rows = []

for i in indices:
    ex = full[i]
    orig_choices = ex["choices"]
    correct_idx  = ex["answer"]
    # pick one random distractor
    distractors = [j for j in range(len(orig_choices)) if j != correct_idx]
    d_idx = rng.choice(distractors)
    # build two-choice list
    two = [orig_choices[correct_idx], orig_choices[d_idx]]
    # shuffle so correct answer isn't always first
    rng.shuffle(two)
    # new correct index
    new_correct = two.index(orig_choices[correct_idx])
    # append row
    rows.append({
        "id":           f"mmlu_{i}",
        "question":     ex["question"],
        "choices":      json.dumps(two),
        "correct_idx":  new_correct,
    })

# -------------------- write TSV
pd.DataFrame(rows).to_csv(subset_path, sep="\t", index=False)
print(f"✅ Wrote  {args.n} items with 2 choices → {subset_path}")
