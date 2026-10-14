"""
Step 2: For sampled training chunks, have Gemini write ONE question and answer it in all audience styles.
One call per chunk; resume-safe (re-run any time, e.g. after the daily free-tier quota resets).

Usage:
    python scripts/data_pipeline/02_generate_qa.py --limit 200      # do up to 200 new chunks this session
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, QuotaExhausted, preflight, append_jsonl, ask_json, gemini_client,  # noqa: E402
                    load_cfg, load_styles, read_jsonl, shared_rules)

QTYPES = [
    "a 'why' question about a cause or reason",
    "a 'how does it work' question about a mechanism",
    "a question about scale, size, distance or time",
    "a question comparing two things or two cases",
    "a 'what is' question for a key concept",
    "a question that corrects a common misconception",
]
SKIP_MARKERS = ("Thought Questions", "Figuring For Yourself", "Review Questions", "Collaborative Group Activities")

PROMPT = """<<RULES>>

You are writing training data for an astronomy tutor. Below is a passage from <<SOURCE>>.

PASSAGE:
<<CHUNK>>

Task:
1. Write ONE self-contained question a learner might ask whose answer is supported by the passage. Make it <<QTYPE>>. It must make sense without the passage: never say "the passage", "the text", "this chapter", or "the author".
2. Answer it three times, each pitched at a different audience:
<<STYLES>>

Every factual claim and number must be supported by the passage (do not add outside facts). Do not mention the passage or the audience. Keep the same warm voice in all three answers; only the depth and vocabulary change.

Return ONLY JSON: {"question": "...", "answers": {<<KEYS>>}}"""


def pick_chunks(chunks, n, seed):
    rng = random.Random(seed)
    good = [c for c in chunks if not any(m.lower() in c["text"].lower() for m in SKIP_MARKERS)]
    by_ch = {}
    for c in good:
        by_ch.setdefault(c["chapter_idx"], []).append(c)
    for v in by_ch.values():
        rng.shuffle(v)
    order, keys = [], sorted(by_ch)
    while len(order) < n and any(by_ch.values()):
        for k in keys:
            if by_ch[k]:
                order.append(by_ch[k].pop())
                if len(order) >= n:
                    break
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max NEW chunks this run")
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    g = cfg["generation"]
    styles = load_styles()
    chunks = read_jsonl(ROOT / "data/corpus/chunks_train.jsonl")
    if not chunks:
        sys.exit("Run 01_extract_corpus.py first.")
    plan = pick_chunks(chunks, g["target_questions"], cfg["seed"])

    out = ROOT / "data/interim/raw_qa.jsonl"
    done = {r["chunk_id"] for r in read_jsonl(out)}
    todo = [(i, c) for i, c in enumerate(plan) if c["chunk_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to do this run (target {len(plan)})")

    style_block = "\n".join(f'   - "{k}" = {v["name"]}: {v["description"]} ({v["words"][0]}-{v["words"][1]} words)'
                            for k, v in styles.items())
    keys = ", ".join(f'"{k}": "..."' for k in styles)
    client = gemini_client()
    preflight(client, g["gen_model"])
    rules = shared_rules()

    for n, (i, c) in enumerate(todo, 1):
        prompt = (PROMPT.replace("<<RULES>>", rules)
                  .replace("<<SOURCE>>", "OpenStax Astronomy 2e" if c["source"].startswith("openstax") else "Wikipedia")
                  .replace("<<CHUNK>>", c["text"])
                  .replace("<<QTYPE>>", QTYPES[i % len(QTYPES)])
                  .replace("<<STYLES>>", style_block)
                  .replace("<<KEYS>>", keys))
        try:
            res = ask_json(client, g["gen_model"], prompt, temperature=g["temperature"])
        except QuotaExhausted as e:
            print(f"\nQuota/rate limit hit after {n-1} chunks this run. Re-run later; progress is saved.\n{str(e)[:200]}")
            return
        ok = (isinstance(res, dict) and isinstance(res.get("question"), str)
              and isinstance(res.get("answers"), dict)
              and all(isinstance(res["answers"].get(k), str) and res["answers"][k].strip() for k in styles))
        if not ok:
            print(f"[{n}/{len(todo)}] {c['chunk_id']}: bad output, skipped")
            continue
        append_jsonl(out, {
            "qid": f"q-{c['chunk_id']}", "question": res["question"].strip(), "answers": res["answers"],
            "chunk_id": c["chunk_id"], "chapter": c["chapter"], "chapter_idx": c["chapter_idx"],
            "source": c["source"], "chunk_text": c["text"], "gen_model": g["gen_model"],
        })
        print(f"[{n}/{len(todo)}] {c['chunk_id']}: {res['question'][:70]}")
        time.sleep(g["sleep_seconds"])
    print("Done with this batch.")


if __name__ == "__main__":
    main()
