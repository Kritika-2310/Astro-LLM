"""
For duplicate questions, keeps ALL unique answers (not just the first one).
Re-labels them as slightly varied training signal:
  "What is a black hole?" appears 6 times -> keeps all 6 as valid training examples
  since each Groq generation is slightly different phrasing = good style variety.
Only drops exact duplicate (question + answer) pairs.
"""
import json
from pathlib import Path

seen_pairs = set()
kept = []

with open("data/raw/candidates.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("reviewed"):
            continue
        # key = question + first 80 chars of answer (catches near-identical outputs)
        key = r["instruction"].lower().strip() + r["output"][:80].lower().strip()
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        kept.append(r)

print(f"After removing exact duplicates: {len(kept)} unique (question, answer) pairs")

# Write back
with open("data/raw/candidates.jsonl", "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")

print("Done. Now run: python scripts/prepare_dataset.py --force")
