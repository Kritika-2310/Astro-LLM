"""
Held-out MCQ accuracy: base vs QLoRA adapter vs RAG vs hybrid (adapter + RAG).
Scores by comparing next-token logits of A/B/C/D (deterministic, no generation, fast on a T4).

  python scripts/eval_mcq.py --name base
  python scripts/eval_mcq.py --name qlora  --adapter /content/drive/MyDrive/<adapter_dir>
  python scripts/eval_mcq.py --name rag    --rag
  python scripts/eval_mcq.py --name hybrid --adapter <adapter_dir> --rag

RAG = TF-IDF top-k over data/corpus/chunks_heldout.jsonl (held-out chapters only). Also reports retrieval hit-rate
(gold chunk in top-k), which is the metric to quote for the retriever on its own.
"""
import argparse, json, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mcq", default="data/eval/heldout_mcq_clean.jsonl")
    ap.add_argument("--chunks", default="data/corpus/chunks_heldout.jsonl")
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args()

    mcqs = read_jsonl(a.mcq)
    retr = None
    if a.rag:
        from sklearn.feature_extraction.text import TfidfVectorizer
        chunks = read_jsonl(a.chunks)
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform([c["text"] for c in chunks])
        def retr(q):
            sims = (mat @ vec.transform([q]).T).toarray().ravel()
            return [chunks[i] for i in sims.argsort()[::-1][:a.k]]

    use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    dt = torch.bfloat16 if use_bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, device_map="auto", token=os.environ.get("HF_TOKEN"),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True))
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    letters = ["A", "B", "C", "D"]
    ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    correct, hits, rows = 0, 0, []
    for m in mcqs:
        ctx = ""
        if retr:
            top = retr(m["question"])
            hits += any(c["chunk_id"] == m["gold_chunk_id"] for c in top)
            ctx = "Context:\n" + "\n\n".join(c["text"] for c in top) + "\n\n"
        opts = "\n".join(f"{k}. {v}" for k, v in m["options"].items())
        user = f"{ctx}Answer this astronomy multiple-choice question with a single letter.\n\n{m['question']}\n{opts}"
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True) + "The answer is "
        x = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**x).logits[0, -1]
        pred = letters[int(torch.stack([logits[i] for i in ids]).argmax())]
        ok = pred == m["answer"]
        correct += ok
        rows.append({"id": m["id"], "pred": pred, "gold": m["answer"], "ok": ok})

    n = len(mcqs)
    res = {"name": a.name, "n": n, "accuracy": round(correct / n, 4), "correct": correct,
           "adapter": a.adapter, "rag": a.rag, "k": a.k if a.rag else None,
           "retrieval_hit_rate": round(hits / n, 4) if a.rag else None}
    Path(a.out_dir).mkdir(exist_ok=True)
    (Path(a.out_dir) / f"mcq_{a.name}.json").write_text(json.dumps({**res, "rows": rows}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
