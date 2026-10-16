"""
Generate answers for the style test: base model vs QLoRA adapter, same prompt (audience template) for both.
Run on Colab (GPU). Greedy decoding, so results are reproducible.

  python scripts/gen_style.py --adapter /content/drive/MyDrive/astro-llm/final_adapter
Writes results/style_answers.jsonl  (one row per question x audience, with base_answer and adapter_answer).
"""
import argparse, json, os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompting import build_prompt, load_styles

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--questions", default="data/processed_v2/test.jsonl")
    ap.add_argument("--template", default="prompts/audience_style_template.txt")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out", default="results/style_answers.jsonl")
    a = ap.parse_args()

    rows = read_jsonl(a.questions)
    styles = load_styles()
    template = Path(a.template).read_text()
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    done = {(r["question"], r["style"]) for r in read_jsonl(out)} if out.exists() else set()

    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, device_map="auto", token=os.environ.get("HF_TOKEN"),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True))
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    eos = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]

    def gen(prompt):
        x = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=a.max_new_tokens, do_sample=False,
                               eos_token_id=[e for e in eos if e is not None], pad_token_id=tok.pad_token_id)
        return tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    todo = [r for r in rows if (r["instruction"], r["style"]) not in done]
    print(f"{len(todo)} to generate ({len(done)} already done)")
    with open(out, "a") as f:
        for i, r in enumerate(todo, 1):
            q = r["instruction"]
            prompt = build_prompt(template, q, r["style"], styles)
            with model.disable_adapter():
                base_ans = gen(prompt)
            ad_ans = gen(prompt)
            f.write(json.dumps({"question": q, "style": r["style"], "reference": r.get("output", ""),
                                "base_answer": base_ans, "adapter_answer": ad_ans}) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}]", flush=True)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
