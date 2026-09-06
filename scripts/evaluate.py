"""
Scores a model against data/eval_benchmark.jsonl two ways:
1. Rule-based fact coverage
2. Gemini as LLM judge (accuracy + voice, 1-5 each)

Usage:
    python scripts/evaluate.py --model-path base --label base_model
    python scripts/evaluate.py --model-path outputs/astro-llm-r16-lr2e4-e3/final_adapter --label finetuned
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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="auto",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN")
    )
    if model_path != "base":
        model = PeftModel.from_pretrained(model, model_path)

    prompt = template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=300, do_sample=True,
        temperature=0.7, top_p=0.9
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return text[len(prompt_text):].strip()

def rule_score(answer, item):
    al = answer.lower()
    def coverage(fact):
        words = [w for w in fact.lower().split() if len(w) > 4]
        if not words:
            return 1.0 if fact.lower() in al else 0.0
        return sum(1 for w in words if w in al) / len(words)
    fact_scores = [coverage(f) for f in item["must_include_facts"]]
    pitfalls = sum(1 for p in item["must_avoid"] if p.lower()[:25] in al)
    return {
        "fact_coverage": sum(fact_scores) / max(len(fact_scores), 1),
        "pitfalls_triggered": pitfalls
    }

def gemini_judge(client, question, answer):
    prompt = f"""You are grading an astronomy explanation on two dimensions.

Question: {question}
Answer: {answer}

Grade on:
1. Scientific accuracy (1-5): Are all facts correct? Any misleading claims?
2. Voice quality (1-5): Is it vivid and engaging like Carl Sagan or Neil deGrasse Tyson?

Reply ONLY with valid JSON, no markdown:
{{"accuracy_score": <1-5>, "voice_score": <1-5>, "reasoning": "<2 sentences>"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"    Gemini error: {e}")
        return {"accuracy_score": None, "voice_score": None, "reasoning": f"ERROR: {e}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-id",
                        default=os.environ.get("BASE_MODEL_ID",
                        "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    template = Path("prompts/sagan_style_template.txt").read_text()
    benchmark = load_benchmark()
    out_path = Path(f"outputs/eval_{args.label}.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    results = []
    for item in benchmark:
        print(f"  Generating answer for [{item['id']}]...")
        answer = generate_answer(
            args.model_path, args.base_model_id,
            item["question"], template
        )
        rb = rule_score(answer, item)
        judge = gemini_judge(client, item["question"], answer)
        result = {**item, "answer": answer, **rb, **judge}
        results.append(result)
        print(f"    fact={rb['fact_coverage']:.2f}  "
              f"pitfalls={rb['pitfalls_triggered']}  "
              f"acc={judge.get('accuracy_score')}  "
              f"voice={judge.get('voice_score')}")
        if judge.get("reasoning"):
            print(f"    Gemini: {judge['reasoning'][:80]}")

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    avg_fact = sum(r["fact_coverage"] for r in results) / n
    total_pitfalls = sum(r["pitfalls_triggered"] for r in results)
    acc = [r["accuracy_score"] for r in results if r.get("accuracy_score")]
    voice = [r["voice_score"] for r in results if r.get("voice_score")]

    print(f"\n=== {args.label} Summary ===")
    print(f"Avg fact coverage : {avg_fact:.2%}")
    print(f"Total pitfalls    : {total_pitfalls}")
    if acc:   print(f"Avg accuracy (Gemini): {sum(acc)/len(acc):.2f}/5")
    if voice: print(f"Avg voice    (Gemini): {sum(voice)/len(voice):.2f}/5")
    print(f"Results -> {out_path}")

if __name__ == "__main__":
    main()
