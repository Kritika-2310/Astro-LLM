"""
Independent factual-error check (no pairing, no model names, so no position/verbosity halo).
Each answer is judged ALONE against the reference answer + the judge's own knowledge: does it contain a factual error
(a claim that contradicts the reference or established science, or an invented mechanism)?

  python scripts/judge_facts.py            # needs GEMINI_API_KEY in .env; resumable (~200 calls)
Reads results/style_answers.jsonl; writes results/fact_judgments.jsonl and results/fact_summary.json
Arms compared: adapter_answer vs base_prompted_answer (change with --arms).
"""
import argparse, json, sys, time
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data_pipeline"))
from common import QuotaExhausted, ask_json, gemini_client, load_cfg, preflight, read_jsonl  # noqa: E402

PROMPT = """You are fact-checking an astronomy answer written for a general audience.

QUESTION: {q}

REFERENCE ANSWER (textbook-grounded):
{ref}

ANSWER TO CHECK:
{a}

Does the ANSWER contain a factual error? Count it as an error if it (1) states something that contradicts the reference
or well-established astronomy/physics, or (2) invents a mechanism or fact that is wrong or unsupported, even if phrased
as an analogy. Do NOT count: simplifications that are still true, harmless analogies that do not distort the physics,
or omitted details. Be strict but fair.
Return ONLY JSON: {{"factual_error": true or false, "quote": "the erroneous sentence, or empty", "why": "max 15 words"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="results/style_answers.jsonl")
    ap.add_argument("--judgments", default="results/fact_judgments.jsonl")
    ap.add_argument("--summary", default="results/fact_summary.json")
    ap.add_argument("--arms", nargs="+", default=["adapter_answer", "base_prompted_answer"])
    ap.add_argument("--sleep", type=float, default=4.0)
    a = ap.parse_args()

    rows = [r for r in read_jsonl(a.answers) if all(k in r for k in a.arms)]
    cfg = load_cfg(); model = cfg["filter"]["judge_model"]
    client = gemini_client(); preflight(client, model)
    jp = Path(a.judgments)
    done = {(j["question"], j["style"], j["arm"]) for j in read_jsonl(jp)}
    try:
        with open(jp, "a") as f:
            for i, r in enumerate(rows, 1):
                for arm in a.arms:
                    if (r["question"], r["style"], arm) in done:
                        continue
                    v = ask_json(client, model, PROMPT.format(q=r["question"], ref=r["reference"], a=r[arm]), temperature=0.0)
                    if not isinstance(v, dict) or "factual_error" not in v:
                        continue
                    rec = {"question": r["question"], "style": r["style"], "arm": arm,
                           "error": bool(v["factual_error"]), "quote": v.get("quote", ""), "why": v.get("why", "")}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    time.sleep(a.sleep)
                print(f"[{i}/{len(rows)}]", flush=True)
    except QuotaExhausted as e:
        print(f"Quota: {e}\nProgress saved; re-run later.")

    js = read_jsonl(jp)
    by = {(j["question"], j["style"], j["arm"]): j["error"] for j in js}
    summary = {"arms": a.arms, "error_rate": {}, "by_style": {}, "paired": {}}
    for arm in a.arms:
        e = [j["error"] for j in js if j["arm"] == arm]
        summary["error_rate"][arm] = {"errors": sum(e), "n": len(e), "pct": round(100 * sum(e) / len(e), 1) if e else None}
        for s in ("beginner", "enthusiast", "advanced"):
            es = [j["error"] for j in js if j["arm"] == arm and j["style"] == s]
            summary["by_style"].setdefault(s, {})[arm] = f"{sum(es)}/{len(es)}"
    if len(a.arms) == 2:
        x, y = a.arms
        only_x = sum(1 for (q, s, arm), v in by.items() if arm == x and v and by.get((q, s, y)) is False)
        only_y = sum(1 for (q, s, arm), v in by.items() if arm == y and v and by.get((q, s, x)) is False)
        n = only_x + only_y
        p = min(1.0, 2 * sum(comb(n, k) for k in range(max(only_x, only_y), n + 1)) / 2 ** n) if n else 1.0
        summary["paired"] = {f"only_{x}_wrong": only_x, f"only_{y}_wrong": only_y, "sign_test_p": round(p, 4)}
    Path(a.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
