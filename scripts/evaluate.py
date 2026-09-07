"""
Scores a model against data/eval_benchmark.jsonl.
1. Rule-based fact coverage
2. Gemini judge — accuracy + voice (1-5 each)
"""
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import gc

load_dotenv()

def load_benchmark():
    with open("data/eval_benchmark.jsonl") as f:
        return [json.loads(l) for l in f]

def generate_answer(model_path, base_model_id, question, template):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="cpu",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN")
    )
    if model_path != "base":
        model = PeftModel.from_pretrained(model, model_path)

    model.to("cuda")
    prompt = template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=True,
            temperature=0.7, top_p=0.9
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    
    # Clean up
    del model, tokenizer, inputs, out
    torch.cuda.empty_cache()
    gc.collect()
    
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
    prompt = f"""Grade this astronomy answer on accuracy (1-5) and voice (1-5).

Question: {question}
Answer: {answer}

Reply ONLY with JSON:
{{"accuracy_score": <1-5>, "voice_score": <1-5>, "reasoning": "<1 sentence>"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"accuracy_score": None, "voice_score": None, "reasoning": f"ERROR"}

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
    for i, item in enumerate(benchmark):
        print(f"[{i+1}/{len(benchmark)}] {item['id']}")
        try:
            answer = generate_answer(
                args.model_path, args.base_model_id,
                item["question"], template
            )
        except Exception as e:
            print(f"  Gen error: {e}")
            continue
            
        rb = rule_score(answer, item)
        judge = gemini_judge(client, item["question"], answer)
        result = {**item, "answer": answer, **rb, **judge}
        results.append(result)
        print(f"  fact={rb['fact_coverage']:.2f} acc={judge.get('accuracy_score')} voice={judge.get('voice_score')}")

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    if n > 0:
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
