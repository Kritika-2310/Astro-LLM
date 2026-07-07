"""
Scores a model against data/eval_benchmark.jsonl two ways:
1. Rule-based fact coverage check
2. LLM-as-judge (Claude) for accuracy + voice quality

On a CPU Codespace: run with --skip-generation and pre-saved answers.
On a GPU machine: run normally to generate + score in one pass.

Usage:
    python scripts/evaluate.py --model-path base --label base_model
    python scripts/evaluate.py --model-path outputs/run/final_adapter --label finetuned
"""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_benchmark():
    with open("data/eval_benchmark.jsonl") as f:
        return [json.loads(l) for l in f]

def generate_answer(model_path, base_model_id, question, template):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto", torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN")
    )
    if model_path != "base":
        model = PeftModel.from_pretrained(model, model_path)

    prompt = template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return text[len(prompt_text):].strip()

def rule_score(answer, item):
    al = answer.lower()
    keywords_hit = lambda fact: sum(1 for w in fact.lower().split() if len(w) > 4 and w in al)
    keywords_total = lambda fact: max(sum(1 for w in fact.lower().split() if len(w) > 4), 1)
    fact_scores = [keywords_hit(f) / keywords_total(f) for f in item["must_include_facts"]]
    pitfalls = sum(1 for p in item["must_avoid"] if p.lower()[:20] in al)
    return {
        "fact_coverage": sum(fact_scores) / max(len(fact_scores), 1),
        "pitfalls_triggered": pitfalls
    }

def judge_score(client, question, answer):
    prompt = f"""Grade this astronomy answer on two dimensions:
1. Scientific accuracy (any factual errors or misleading claims?)
2. Voice quality (vivid, warm, engaging like Sagan/deGrasse Tyson without sacrificing accuracy?)

Question: {question}
Answer: {answer}

Reply ONLY with JSON (no markdown): {{"accuracy_score": <1-5>, "voice_score": <1-5>, "reasoning": "<2 sentences>"}}"""
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(text)
    except:
        return {"accuracy_score": None, "voice_score": None, "reasoning": f"PARSE_ERROR: {text}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-id", default=os.environ.get("BASE_MODEL_ID","meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    template = Path("prompts/sagan_style_template.txt").read_text()
    benchmark = load_benchmark()
    out_path = Path(f"outputs/eval_{args.label}.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    results = []
    for item in benchmark:
        answer = generate_answer(args.model_path, args.base_model_id, item["question"], template)
        rb = rule_score(answer, item)
        judge = judge_score(client, item["question"], answer)
        result = {**item, "answer": answer, **rb, **judge}
        results.append(result)
        print(f"[{item['id']}] fact={rb['fact_coverage']:.2f} pitfalls={rb['pitfalls_triggered']} acc={judge.get('accuracy_score')} voice={judge.get('voice_score')}")

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    print(f"\n=== {args.label} summary ===")
    print(f"Avg fact coverage: {sum(r['fact_coverage'] for r in results)/n:.2%}")
    print(f"Total pitfalls: {sum(r['pitfalls_triggered'] for r in results)}")
    acc_scores = [r['accuracy_score'] for r in results if r.get('accuracy_score')]
    voice_scores = [r['voice_score'] for r in results if r.get('voice_score')]
    if acc_scores: print(f"Avg accuracy score: {sum(acc_scores)/len(acc_scores):.2f}/5")
    if voice_scores: print(f"Avg voice score: {sum(voice_scores)/len(voice_scores):.2f}/5")
    print(f"Results -> {out_path}")

if __name__ == "__main__":
    main()
