"""
Style test scoring: base vs adapter answers.
  1. Objective (no API): % of answers inside the audience's word range, mean length per audience,
     and whether length rises beginner < enthusiast < advanced (does the model actually adapt?).
  2. Blind pairwise LLM judge (Gemini): audience fit, accuracy vs the reference answer, overall.
     Answer order is randomised per item so position bias doesn't favour either model.

  python scripts/judge_style.py            # needs GEMINI_API_KEY in .env; resumable
Reads results/style_answers.jsonl, writes results/style_judgments.jsonl and results/style_summary.json
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data_pipeline"))
from common import (QuotaExhausted, ask_json, gemini_client, load_cfg, load_styles, preflight,  # noqa: E402
                    read_jsonl, word_count)

PROMPT = """You are grading two answers (A and B) to an astronomy question written for a specific audience.

AUDIENCE: {name}. {desc}

QUESTION: {q}

REFERENCE ANSWER (textbook-grounded; use it to check facts):
{ref}

ANSWER A:
{a}

ANSWER B:
{b}

For each criterion pick "A", "B" or "tie":
- audience_fit: which is better pitched for this audience (vocabulary, depth, analogies, length)?
- accuracy: which is more scientifically accurate and free of invented or wrong claims (compare to the reference)?
- overall: which would you rather give this audience?
Return ONLY JSON: {{"audience_fit": "A|B|tie", "accuracy": "A|B|tie", "overall": "A|B|tie"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="results/style_answers.jsonl")
    ap.add_argument("--judgments", default="results/style_judgments.jsonl")
    ap.add_argument("--summary", default="results/style_summary.json")
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--no_judge", action="store_true", help="only the objective length metrics")
    a = ap.parse_args()

    rows = read_jsonl(a.answers)
    styles = load_styles()
    summary = {"n": len(rows), "length": {}, "judge": {}}

    # ---- 1. objective length metrics
    for who in ("base", "adapter"):
        per = defaultdict(list)
        for r in rows:
            per[r["style"]].append(word_count(r[f"{who}_answer"]))
        inr = sum(styles[r["style"]]["words"][0] <= word_count(r[f"{who}_answer"]) <= styles[r["style"]]["words"][1]
                  for r in rows)
        means = {s: round(sum(v) / len(v), 1) for s, v in per.items()}
        order = [means.get(s) for s in ("beginner", "enthusiast", "advanced")]
        summary["length"][who] = {"mean_words": means, "in_range": f"{inr}/{len(rows)}",
                                  "in_range_pct": round(100 * inr / len(rows), 1),
                                  "length_rises_with_level": all(x is not None for x in order)
                                  and order[0] < order[1] < order[2]}

    # ---- 2. blind pairwise judge
    if not a.no_judge:
        cfg = load_cfg()
        model = cfg["filter"]["judge_model"]
        client = gemini_client()
        preflight(client, model)
        jp = Path(a.judgments)
        done = {(j["question"], j["style"]) for j in read_jsonl(jp)}
        rng = random.Random(0)
        try:
            with open(jp, "a") as f:
                for i, r in enumerate(rows, 1):
                    flip = rng.random() < 0.5            # decided for every row so reruns stay consistent
                    if (r["question"], r["style"]) in done:
                        continue
                    A, B = (r["adapter_answer"], r["base_answer"]) if flip else (r["base_answer"], r["adapter_answer"])
                    s = styles[r["style"]]
                    v = ask_json(client, model, PROMPT.format(name=s["name"], desc=s["description"], q=r["question"],
                                                              ref=r["reference"], a=A, b=B), temperature=0.0)
                    if not v:
                        continue
                    unflip = {"A": "adapter" if flip else "base", "B": "base" if flip else "adapter", "tie": "tie"}
                    rec = {"question": r["question"], "style": r["style"],
                           **{k: unflip.get(str(v.get(k)).strip(), "tie") for k in ("audience_fit", "accuracy", "overall")}}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"[{i}/{len(rows)}] {rec['overall']}", flush=True)
                    time.sleep(a.sleep)
        except QuotaExhausted as e:
            print(f"Quota: {e}\nProgress saved; re-run later.")
        js = read_jsonl(jp)
        summary["judge"]["n_judged"] = len(js)
        for k in ("audience_fit", "accuracy", "overall"):
            c = {w: sum(j[k] == w for j in js) for w in ("adapter", "base", "tie")}
            decisive = c["adapter"] + c["base"]
            # two-sided exact sign test on decisive items
            from math import comb
            p = 1.0
            if decisive:
                kk = max(c["adapter"], c["base"])
                p = min(1.0, 2 * sum(comb(decisive, x) for x in range(kk, decisive + 1)) / 2 ** decisive)
            summary["judge"][k] = {**c, "adapter_win_rate_of_decisive": round(c["adapter"] / decisive, 3) if decisive else None,
                                   "sign_test_p": round(p, 4)}
        by = defaultdict(lambda: defaultdict(int))
        for j in js:
            by[j["style"]][j["overall"]] += 1
        summary["judge"]["overall_by_style"] = {s: dict(v) for s, v in by.items()}

    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
