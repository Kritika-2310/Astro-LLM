"""
Diffs base vs finetuned eval outputs. Prints per-category table and flags regressions.

Usage:
    python scripts/compare_results.py
"""
import json
from collections import defaultdict
from pathlib import Path

def load(path):
    with open(path) as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f}

base = load("outputs/eval_base_model.jsonl")
ft = load("outputs/eval_finetuned.jsonl")

per_cat = defaultdict(lambda: {"ba":[],"fa":[],"bv":[],"fv":[]})
improvements, regressions = [], []

for id_, b in base.items():
    f = ft.get(id_)
    if not f: continue
    cat = b["category"]
    per_cat[cat]["ba"].append(b.get("accuracy_score") or 0)
    per_cat[cat]["fa"].append(f.get("accuracy_score") or 0)
    per_cat[cat]["bv"].append(b.get("voice_score") or 0)
    per_cat[cat]["fv"].append(f.get("voice_score") or 0)
    cb = (b.get("accuracy_score") or 0) + (b.get("voice_score") or 0)
    cf = (f.get("accuracy_score") or 0) + (f.get("voice_score") or 0)
    entry = {"id": id_, "question": b["question"], "delta": cf - cb}
    (improvements if cf > cb else regressions if cf < cb else []).append(entry)

print("=== Per-category (base -> finetuned) ===")
for cat, v in per_cat.items():
    n = len(v["ba"])
    print(f"{cat:25s}  accuracy: {sum(v['ba'])/n:.2f} -> {sum(v['fa'])/n:.2f}  voice: {sum(v['bv'])/n:.2f} -> {sum(v['fv'])/n:.2f}")

print(f"\nImprovements: {len(improvements)}  |  Regressions: {len(regressions)}")
if regressions:
    print("\nRegressions (include these honestly in your report):")
    for r in regressions:
        print(f"  {r['id']} (delta={r['delta']:.1f}): {r['question']}")

report = {"improvements": improvements, "regressions": regressions, "per_category": {k: dict(v) for k,v in per_cat.items()}}
Path("outputs/comparison_report.json").write_text(json.dumps(report, indent=2))
print("\nFull report -> outputs/comparison_report.json")
