"""
Generates CANDIDATE training examples using Groq (Llama 3 70B).
Fast, free tier, no Anthropic key needed.

Every example is written with reviewed=false.
YOU must open data/raw/candidates.jsonl, fact-check each answer
against nasa.gov / OpenStax Astronomy, edit if needed, and flip
reviewed=true before running prepare_dataset.py.

Usage:
    python scripts/generate_dataset.py --n 50
    python scripts/generate_dataset.py --n 100 --topics-file data/raw/topics.txt
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"   # best Groq model for instruction following

def load_template():
    return Path("prompts/sagan_style_template.txt").read_text()

def generate_one(client, question, template):
    prompt = template.format(question=question)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--topics-file", default="data/raw/topics.txt")
    parser.add_argument("--out", default="data/raw/candidates.jsonl")
    args = parser.parse_args()

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    template = load_template()
    topics = [l.strip() for l in Path(args.topics_file).read_text().splitlines() if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    with out_path.open("a") as f:
        while generated < args.n:
            question = topics[generated % len(topics)]
            try:
                answer = generate_one(client, question, template)
            except Exception as e:
                print(f"  Error on '{question}': {e}")
                time.sleep(3)
                continue

            record = {
                "instruction": question,
                "input": "",
                "output": answer,
                "reviewed": False,
                "source": f"synthetic_groq_{GROQ_MODEL}"
            }
            f.write(json.dumps(record) + "\n")
            generated += 1
            print(f"[{generated}/{args.n}] {question[:65]}")
            time.sleep(0.3)   # stay within Groq free tier rate limits

    print(f"\nDone. {generated} candidates -> {out_path}")
    print("\nNext steps:")
    print("  1. Open data/raw/candidates.jsonl in VS Code")
    print("  2. Fact-check every answer against nasa.gov or OpenStax Astronomy")
    print("  3. Edit any inaccurate lines, flip reviewed=false to reviewed=true")
    print("  4. Once you have 500+ reviewed examples, run: python scripts/prepare_dataset.py")

if __name__ == "__main__":
    main()
