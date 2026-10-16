"""
Fixes the truncation confound in the style test. Re-generates ONLY the base-model answers, in two arms, and adds them
to results/style_answers.jsonl (adapter answers are reused; they are greedy and none were truncated):

  base_long_answer      base model, same prompt, but max_new_tokens=600 so answers can finish
  base_prompted_answer  base model, same prompt + an explicit word range ("Write 40-130 words.") per audience
                        = the "just prompt it better" baseline the fine-tune has to beat

  python scripts/gen_base_arms.py
"""
import argparse, json, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompting import build_prompt, load_styles

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="results/style_answers.jsonl")
    ap.add_argument("--template", default="prompts/audience_style_template.txt")
    ap.add_argument("--max_new_tokens", type=int, default=600)
    a = ap.parse_args()

    p = Path(a.answers)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    styles = load_styles()
    prompted = {k: {**v, "description": f"{v['description']} Write {v['words'][0]}-{v['words'][1]} words."}
                for k, v in styles.items()}
    template = Path(a.template).read_text()

    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN"))
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dt, device_map="auto", token=os.environ.get("HF_TOKEN"),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True))
    model.eval()
    eos = [e for e in (tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")) if e is not None]

    def gen(prompt):
        x = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=a.max_new_tokens, do_sample=False,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        t = tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return t.split("\nQuestion:")[0].strip()      # drop any run-on of a new made-up question

    todo = [r for r in rows if "base_prompted_answer" not in r]
    print(f"{len(todo)} rows to do")
    for i, r in enumerate(todo, 1):
        r["base_long_answer"] = gen(build_prompt(template, r["question"], r["style"], styles))
        r["base_prompted_answer"] = gen(build_prompt(template, r["question"], r["style"], prompted))
        if i % 10 == 0 or i == len(todo):
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")     # checkpoint (resumable)
            print(f"[{i}/{len(todo)}]", flush=True)
    for k in ("base_long_answer", "base_prompted_answer"):
        cut = sum(not r[k].rstrip().endswith((".", "!", "?", '"', ")")) for r in rows if k in r)
        print(f"{k}: ends mid-sentence {cut}/{len(rows)}")
    print("Done")


if __name__ == "__main__":
    main()
