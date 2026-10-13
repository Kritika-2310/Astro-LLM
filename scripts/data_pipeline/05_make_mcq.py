"""
Step 5: Build a held-out-chapter MCQ set (for base vs fine-tuned vs RAG vs hybrid comparison).
Questions come ONLY from held-out chapters, so the fine-tuned model never saw that material.
Each MCQ keeps `gold_chunk_id`, so you can also measure RAG retrieval hit-rate.

Verification: a second call answers the MCQ from the passage alone; kept only if it matches the key.
Then YOU review every item (set "reviewed": true) -- the set is small, errors here corrupt headline numbers.

Usage:
    python scripts/data_pipeline/05_make_mcq.py --limit 80
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, QuotaExhausted, preflight, append_jsonl, ask_json, gemini_client,  # noqa: E402
                    load_cfg, read_jsonl)

GEN = """From the passage below write ONE multiple-choice question that tests a specific fact or concept.
Rules: self-contained (never say "the passage"/"the text"); exactly one correct answer; three plausible but
clearly wrong distractors; no "all of the above"/"none of the above".

PASSAGE:
<<CHUNK>>

Return ONLY JSON: {"question": "...", "correct": "...", "distractors": ["...", "...", "..."]}"""

VERIFY = """Answer using ONLY the passage. Reply ONLY with JSON {"answer": "A"} (a single letter).

PASSAGE:
<<CHUNK>>

QUESTION: <<Q>>
A) <<A>>
B) <<B>>
C) <<C>>
D) <<D>>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    rng = random.Random(cfg["seed"])
    chunks = read_jsonl(ROOT / "data/corpus/chunks_heldout.jsonl")
    if not chunks:
        sys.exit("No held-out chunks. Run step 1 first.")
    rng.shuffle(chunks)
    plan = chunks[: cfg["mcq"]["n_questions"]]
    out = ROOT / cfg["mcq"]["out_path"]
    seen = {r["gold_chunk_id"] for r in read_jsonl(out)}
    rej_path = ROOT / "data/interim/mcq_rejected.txt"
    if rej_path.exists():
        seen |= {l.strip() for l in rej_path.read_text().splitlines() if l.strip()}
    todo = [c for c in plan if c["chunk_id"] not in seen]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} chunks to process")

    client = gemini_client()
    gen_model, ver_model = cfg["generation"]["gen_model"], cfg["filter"]["judge_model"]
    preflight(client, gen_model, ver_model)
    rejected = rej_path
    for n, c in enumerate(todo, 1):
        try:
            g = ask_json(client, gen_model, GEN.replace("<<CHUNK>>", c["text"]))
            ok = isinstance(g, dict) and isinstance(g.get("distractors"), list) and len(g["distractors"]) == 3 and g.get("correct")
            if not ok:
                continue
            opts = [g["correct"]] + list(g["distractors"])
            rng.shuffle(opts)
            letters = "ABCD"
            key = letters[opts.index(g["correct"])]
            v = ask_json(client, ver_model, VERIFY.replace("<<CHUNK>>", c["text"]).replace("<<Q>>", g["question"])
                         .replace("<<A>>", opts[0]).replace("<<B>>", opts[1]).replace("<<C>>", opts[2]).replace("<<D>>", opts[3]),
                         temperature=0.0)
        except QuotaExhausted as e:
            print(f"\nQuota/rate limit hit after {n-1}. Re-run later; progress is saved.\n{str(e)[:200]}")
            return
        if not isinstance(v, dict) or str(v.get("answer", "")).strip().upper()[:1] != key:
            rejected.parent.mkdir(parents=True, exist_ok=True)
            with rejected.open("a") as f:
                f.write(c["chunk_id"] + "\n")
            print(f"[{n}/{len(todo)}] {c['chunk_id']}: verification mismatch, rejected")
            continue
        append_jsonl(out, {"id": f"mcq-{c['chunk_id']}", "question": g["question"].strip(),
                           "options": dict(zip(letters, opts)), "answer": key,
                           "gold_chunk_id": c["chunk_id"], "chapter": c["chapter"], "source": c["source"],
                           "reviewed": False})
        print(f"[{n}/{len(todo)}] {c['chunk_id']}: ok")
        time.sleep(cfg["generation"]["sleep_seconds"])
    print(f"Done. Now review {out} by hand and set reviewed=true on items you keep.")


if __name__ == "__main__":
    main()
