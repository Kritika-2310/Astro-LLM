"""
Step 4: Split by QUESTION (grouped by source chunk), never by answer.
All styles of a question stay in the same split, and no source chunk is shared across splits.
Writes data/processed_v2/{train,val,test}.jsonl in the same instruction/input/output schema as v1, plus `style`.
Your v1 data in data/processed/ is never touched.

Usage:
    python scripts/data_pipeline/04_split.py            # refuses to overwrite an existing v2 test set
    python scripts/data_pipeline/04_split.py --force
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_cfg, load_styles, read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()

    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    cfg = load_cfg(args.cfg)
    sp = cfg["split"]
    styles = load_styles()
    out_dir = ROOT / sp["out_dir"]
    if (out_dir / "test.jsonl").exists() and not args.force:
        sys.exit(f"{out_dir}/test.jsonl exists. Pass --force to regenerate (destroys the v2 test set).")

    heldout = {h["idx"] for h in json.loads((ROOT / "data/corpus/heldout_chapters.json").read_text())["heldout"]}
    recs = read_jsonl(ROOT / "data/interim/filtered.jsonl")
    if not recs:
        sys.exit("data/interim/filtered.jsonl is empty. Finish step 3 first.")
    assert not any(r["chapter_idx"] in heldout for r in recs), "held-out chapter leaked into filtered data"

    rows = []
    for r in recs:
        for s in styles:
            rows.append({"instruction": r["question"], "input": "", "output": r["answers"][s].strip(),
                         "style": s, "qid": r["qid"], "chunk_id": r["chunk_id"],
                         "chapter": r["chapter"], "source": r["source"]})
    df = pd.DataFrame(rows)

    seed = cfg["seed"]
    g1 = GroupShuffleSplit(n_splits=1, test_size=sp["val"] + sp["test"], random_state=seed)
    tr_i, rest_i = next(g1.split(df, groups=df["chunk_id"]))
    train, rest = df.iloc[tr_i], df.iloc[rest_i]
    g2 = GroupShuffleSplit(n_splits=1, test_size=sp["test"] / (sp["val"] + sp["test"]), random_state=seed)
    va_i, te_i = next(g2.split(rest, groups=rest["chunk_id"]))
    val, test = rest.iloc[va_i], rest.iloc[te_i]

    for a, b, name in ((train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")):
        assert not set(a.qid) & set(b.qid), f"question leakage {name}"
        assert not set(a.chunk_id) & set(b.chunk_id), f"chunk leakage {name}"

    for name, d in (("train", train), ("val", val), ("test", test)):
        write_jsonl(out_dir / f"{name}.jsonl", d.to_dict("records"))
    manifest = {"seed": seed, "questions": {n: int(d.qid.nunique()) for n, d in (("train", train), ("val", val), ("test", test))},
                "rows": {n: len(d) for n, d in (("train", train), ("val", val), ("test", test))},
                "styles": list(styles), "heldout_chapter_idx": sorted(heldout)}
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
