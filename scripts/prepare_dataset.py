"""
Splits reviewed candidates into train/val/test (80/10/10).
The test set is never overwritten once created (it's sacred).
"""
import argparse
import json
import random
from pathlib import Path

def normalize(q):
    return " ".join(q.lower().strip().split())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/raw/candidates.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.jsonl"

    if test_path.exists() and not args.force:
        print("test.jsonl already exists. Pass --force to regenerate (destroys your sacred test set).")
        return

    records, seen = [], set()
    with open(args.candidates) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("reviewed", False):
                continue
            key = normalize(r["instruction"])
            if key in seen:
                continue
            seen.add(key)
            records.append(r)

    print(f"{len(records)} reviewed examples loaded.")
    if len(records) < 10:
        print("Need more reviewed examples. Keep generating and reviewing.")
        return

    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {"train": records[:n_train], "val": records[n_train:n_train+n_val], "test": records[n_train+n_val:]}

    for name, split in splits.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in split:
                f.write(json.dumps({"instruction": r["instruction"], "input": r.get("input",""), "output": r["output"]}) + "\n")
        print(f"{name}: {len(split)} examples")

if __name__ == "__main__":
    main()
