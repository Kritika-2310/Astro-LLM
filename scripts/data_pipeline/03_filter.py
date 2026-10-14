"""
Step 3: Filter generated Q&A.
  a) rule filters (banned phrases, length per style, unsupported numbers, near-identical styles)
  b) question dedupe + leakage check against your existing eval benchmark and old splits
  c) LLM judge (separate model from the generator); a question survives only if ALL styles pass

Usage:
    python scripts/data_pipeline/03_filter.py --skip-judge     # rules + dedupe only (no API calls)
    python scripts/data_pipeline/03_filter.py --limit 150      # judge up to 150 new questions this run (resume-safe)
    python scripts/data_pipeline/03_filter.py                  # finish judging, write filtered.jsonl + report
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, QuotaExhausted, preflight, append_jsonl, ask_json, gemini_client,  # noqa: E402
                    load_cfg, load_styles, read_jsonl, sentences, word_count, write_jsonl)

BANNED = re.compile(r"\b(the passage|the text|this passage|according to the|the author|the chapter|this chapter)\b", re.I)
NUM = re.compile(r"\d[\d,]*\.?\d*")


def norm_nums(s):
    out = set()
    for m in NUM.findall(s):
        m = m.replace(",", "").rstrip(".")
        if m:
            out.add(m)
    return out


def big_numbers(s):
    res = set()
    for m in norm_nums(s):
        try:
            v = float(m)
        except ValueError:
            continue
        if v >= 100 or "." in m:
            res.add(m)
    return res


def jaccard(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(len(A | B), 1)


def rule_check(rec, styles, cfg):
    reasons = []
    if not isinstance(rec.get("question"), str) or not isinstance(rec.get("answers"), dict):
        return ["malformed_record"]
    bad = [k for k in styles if not isinstance(rec["answers"].get(k), str) or not rec["answers"][k].strip()]
    if bad:
        return [f"{k}:not_string" for k in bad]   # stop here; later checks assume strings
    if BANNED.search(rec["question"]):
        reasons.append("question_mentions_source")
    src_nums = norm_nums(rec["chunk_text"])
    for k in styles:
        a = rec["answers"][k]
        lo, hi = styles[k]["words"]
        wc = word_count(a)
        if wc < lo or wc > hi:
            reasons.append(f"{k}:length_{wc}")
        if BANNED.search(a):
            reasons.append(f"{k}:mentions_source")
        if cfg["strict_numbers"] and (big_numbers(a) - src_nums):
            reasons.append(f"{k}:unsupported_number")
    keys = list(styles)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if jaccard(rec["answers"][keys[i]], rec["answers"][keys[j]]) > cfg["min_style_overlap_jaccard"]:
                reasons.append(f"styles_too_similar:{keys[i]}/{keys[j]}")
    return reasons


def reference_questions():
    refs = [r["question"] for r in read_jsonl(ROOT / "data/eval_benchmark.jsonl")]
    for name in ("val", "test"):   # v1 val/test must never resemble v2 training questions
        refs += [r["instruction"] for r in read_jsonl(ROOT / f"data/processed/{name}.jsonl")]
    return refs


def dedupe(recs, refs, threshold, leak_threshold):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not recs:
        return [], 0, 0
    qs = [r["question"] for r in recs]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(qs + refs)
    Q = vec.transform(qs)
    leak = 0
    if refs:
        sim = cosine_similarity(Q, vec.transform(refs)).max(axis=1)
        keep_mask = sim <= leak_threshold
        leak = int((~keep_mask).sum())
    else:
        keep_mask = [True] * len(recs)
    kept_idx, dup = [], 0
    S = cosine_similarity(Q)
    for i in range(len(recs)):
        if not keep_mask[i]:
            continue
        if any(S[i, j] > threshold for j in kept_idx):
            dup += 1
            continue
        kept_idx.append(i)
    return [recs[i] for i in kept_idx], dup, leak


JUDGE = """You are a strict astronomy fact-checker and writing-level rater.

PASSAGE (ground truth):
<<CHUNK>>

QUESTION: <<Q>>

Audiences:
<<AUD>>

Rate each answer from 1 to 5 on:
- faithfulness: every claim is supported by the passage AND scientifically correct (1 = contains an unsupported or wrong claim)
- audience_fit: how well vocabulary, depth and length suit the stated audience
- clarity: clear, well-formed prose without filler

<<ANSWERS>>

Return ONLY JSON: {<<SCHEMA>>}"""


def judge_one(client, model, rec, styles):
    aud = "\n".join(f'- {k}: {v["name"]}. {v["description"]}' for k, v in styles.items())
    answers = "\n\n".join(f"ANSWER ({k}):\n{rec['answers'][k]}" for k in styles)
    schema = ", ".join(f'"{k}": {{"faithfulness": 0, "audience_fit": 0, "clarity": 0}}' for k in styles)
    prompt = (JUDGE.replace("<<CHUNK>>", rec["chunk_text"]).replace("<<Q>>", rec["question"])
              .replace("<<AUD>>", aud).replace("<<ANSWERS>>", answers).replace("<<SCHEMA>>", schema))
    return ask_json(client, model, prompt, temperature=0.0)


def judge_pass(scores, styles, min_score):
    try:
        return all(int(scores[k][m]) >= min_score for k in styles for m in ("faithfulness", "audience_fit", "clarity"))
    except (KeyError, TypeError, ValueError):
        return False


def readability(recs, styles):
    out = {}
    for k in styles:
        txt = [r["answers"][k] for r in recs]
        words = [w for t in txt for w in re.findall(r"[A-Za-z']+", t)]
        sents = [s for t in txt for s in sentences(t)]
        out[k] = {"mean_words_per_answer": round(sum(word_count(t) for t in txt) / max(len(txt), 1), 1),
                  "mean_words_per_sentence": round(len(words) / max(len(sents), 1), 1),
                  "mean_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    f = cfg["filter"]
    styles = load_styles()
    raw = read_jsonl(ROOT / "data/interim/raw_qa.jsonl")
    held_path = ROOT / "data/corpus/heldout_chapters.json"
    held = {h["idx"] for h in json.loads(held_path.read_text())["heldout"]} if held_path.exists() else set()
    n_raw = len(raw)
    raw = [r for r in raw if r["chapter_idx"] not in held]   # safety: held-out chapters never reach training data
    report = {"generated": n_raw, "dropped_heldout_chapter": n_raw - len(raw)}

    ruled, rule_drops = [], {}
    for r in raw:
        reasons = rule_check(r, styles, f)
        if reasons:
            for x in {x.split(":")[1] if ":" in x else x for x in reasons}:
                rule_drops[x] = rule_drops.get(x, 0) + 1
        else:
            ruled.append(r)
    report["after_rules"] = len(ruled)
    report["rule_drop_reasons"] = rule_drops

    deduped, dup, leak = dedupe(ruled, reference_questions(), f["dedupe_threshold"], f["leak_threshold"])
    report.update({"dropped_duplicate_questions": dup, "dropped_similar_to_eval_or_old_splits": leak,
                   "after_dedupe": len(deduped)})

    if args.skip_judge:
        print(json.dumps(report, indent=2))
        print("\n--skip-judge: nothing written. Run without it to judge and write filtered.jsonl.")
        return

    judged_path = ROOT / "data/interim/judged.jsonl"
    judged = {r["qid"]: r for r in read_jsonl(judged_path)}
    todo = [r for r in deduped if r["qid"] not in judged]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(judged)} already judged, {len(todo)} to judge this run")
    client = gemini_client()
    preflight(client, f["judge_model"])
    for n, r in enumerate(todo, 1):
        try:
            sc = judge_one(client, f["judge_model"], r, styles)
        except QuotaExhausted as e:
            print(f"\nQuota/rate limit hit after {n-1} this run. Re-run later; progress is saved.\n{str(e)[:200]}")
            break
        rec = {"qid": r["qid"], "scores": sc, "judge_model": f["judge_model"]}
        append_jsonl(judged_path, rec)
        judged[r["qid"]] = rec
        print(f"[{n}/{len(todo)}] {r['qid']}")
        time.sleep(cfg["generation"]["sleep_seconds"])

    if any(r["qid"] not in judged for r in deduped):
        print("\nNot everything is judged yet; re-run to continue. (No filtered.jsonl written.)")
        return

    final = []
    for r in deduped:
        sc = judged[r["qid"]]["scores"]
        if sc and judge_pass(sc, styles, f["judge_min_score"]):
            final.append({**r, "judge_scores": sc})
    report["after_judge"] = len(final)
    report["judge_pass_rate_of_judged"] = round(len(final) / max(len(deduped), 1), 3)
    report["readability_by_style"] = readability(final, styles)
    report["by_source"] = {s: sum(1 for r in final if r["source"] == s) for s in {r["source"] for r in final}}
    write_jsonl(ROOT / "data/interim/filtered.jsonl", final)
    (ROOT / "data/interim/filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nWrote data/interim/filtered.jsonl and filter_report.json. Next: hand-audit 50 rows (docs/DATA_PIPELINE.md).")


if __name__ == "__main__":
    main()
