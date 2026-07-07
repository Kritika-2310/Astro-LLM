"""
Generates CANDIDATE training examples using Claude.
Every example is written with reviewed=false.
YOU must open data/raw/candidates.jsonl, fact-check each answer,
edit if needed, and flip reviewed to true before running prepare_dataset.py.
"""
import argparse
import json
import os
import time
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def load_template():
    return Path("prompts/sagan_style_template.txt").read_text()

def generate_one(client, question, template):
    prompt = template.format(question=question)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--topics-file", type=str, default="data/raw/topics.txt")
    parser.add_argument("--out", type=str, default="data/raw/candidates.jsonl")
    args = parser.parse_args()

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
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
                print(f"Error on '{question}': {e}")
                time.sleep(2)
                continue
            record = {
                "instruction": question,
                "input": "",
                "output": answer,
                "reviewed": False,
                "source": "synthetic_claude"
            }
            f.write(json.dumps(record) + "\n")
            generated += 1
            print(f"[{generated}/{args.n}] {question[:60]}")
            time.sleep(0.4)

    print(f"\nDone. {generated} candidates -> {out_path}")
    print("Next: open that file, fact-check every answer, set reviewed=true, then run prepare_dataset.py")

if __name__ == "__main__":
    main()
