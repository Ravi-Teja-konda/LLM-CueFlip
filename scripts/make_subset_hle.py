#!/usr/bin/env python3
"""
Create a random N-item TSV subset from the HLE parquet data,
keeping only pure text multiple-choice questions with all original options.
Also prints a few examples for verification.
"""
import argparse
import random
import json
import pandas as pd
from pathlib import Path
import yaml
import re

def main():
    # 1) Parse command-line args
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=200,
                   help="Number of items to sample")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--parquet", type=str,
                   default="scripts/hle/data/test-00000-of-00001.parquet",
                   help="Path to the HLE parquet file")
    args = p.parse_args()

    # 2) Load paths.yaml to know where to write the TSV
    paths = yaml.safe_load(open("configs/paths.yaml"))
    out_path = Path(paths["subset_tsv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) Read the parquet file into a DataFrame
    print(f"📥 Loading HLE data from parquet: {args.parquet} …")
    df = pd.read_parquet(args.parquet)
    print(f"🔍 Loaded {len(df)} rows from parquet")

    # 4) Parse and filter MCQs
    def parse_mcq(row):
        q = row.question
        # find Answer Choices block
        m = re.search(r"Answer Choices:\s*\n((?:[A-Z]\.\s*.*\n?)+)", q)
        if not m:
            return None
        block = m.group(1).strip().splitlines()
        opts = []
        for line in block:
            m2 = re.match(r"([A-Z])\.\s*(.+)$", line)
            if m2:
                opts.append((m2.group(1), m2.group(2)))
        if len(opts) < 2:
            return None

        answer = row.answer
        # map letter to index
        if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
            letter = answer.upper()
            idx = next((i for i,(L,_) in enumerate(opts) if L == letter), None)
        else:
            idx = next((i for i,(_,text) in enumerate(opts) if text == answer), None)
        if idx is None:
            return None

        stem = q.split("Answer Choices:")[0].strip()
        return {
            "id":          row.id,
            "question":    stem,
            "choices":     [text for _,text in opts],
            "correct_idx": idx
        }

    print("🔍 Parsing and filtering text MCQs …")
    parsed = []
    for r in df.itertuples():
        out = parse_mcq(r)
        if out is not None:
            parsed.append(out)
    total = len(parsed)
    print(f"✅ {total} items remain after parsing/filtering")

    # 5) Sample N items without replacement
    rng = random.Random(args.seed)
    if args.n > total:
        raise ValueError(f"Requested {args.n} items, but only {total} available.")
    indices = rng.sample(range(total), args.n)
    print(f"🔢 Sampling {args.n} items (seed={args.seed}) …")

    # 6) Build rows with all choices
    rows = []
    for i in indices:
        ex = parsed[i]
        rows.append({
            "id":          ex["id"],
            "question":    ex["question"],
            "choices":     json.dumps(ex["choices"]),
            "correct_idx": ex["correct_idx"],
        })

    # 7) Write out the TSV
    out_df = pd.DataFrame(rows, columns=["id", "question", "choices", "correct_idx"])
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"✅ Wrote {len(out_df)} items → {out_path}")

if __name__ == "__main__":
    main()
