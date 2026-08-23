"""
Pulls APOD entries from NASA API and generates Sagan-style Q&A pairs
using Gemini (more reliable than Groq free tier).

Usage:
    python scripts/generate_from_nasa.py --n 200
"""
import argparse
import json
import os
import time
import random
from pathlib import Path
from datetime import date, timedelta

import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

NASA_API_KEY = os.environ["NASA_API_KEY"]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

APOD_START = date(1995, 6, 16)
APOD_END = date(2025, 12, 31)


def random_date():
    delta = APOD_END - APOD_START
    return APOD_START + timedelta(days=random.randint(0, delta.days))


def fetch_apod(target_date: date):
    url = "https://api.nasa.gov/planetary/apod"
    resp = requests.get(url, params={
        "api_key": NASA_API_KEY,
        "date": target_date.strftime("%Y-%m-%d"),
    }, timeout=10)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("media_type") != "image":
        return None
    if len(data.get("explanation", "")) < 150:
        return None
    return data


def generate_qa(model, title, explanation):
    prompt = f"""You are an astronomy communicator in the tradition of Carl Sagan and Neil deGrasse Tyson.

Below is a factually accurate explanation from NASA about "{title}".
Use ONLY the facts stated here — do not invent or add anything beyond what is written.

NASA explanation:
{explanation}

Your task:
1. Write ONE clear question a curious person might ask about this topic.
   It must be answerable from the explanation above.
2. Write a vivid, engaging answer in the Sagan/deGrasse Tyson style:
   - Use a concrete analogy to ground the physics
   - Warm tone, sense of awe, flowing prose (no bullet points)
   - Strictly accurate to the NASA text
   - 3-5 sentences
   - Note briefly where an analogy breaks down if needed

Reply ONLY with valid JSON, no markdown:
{{"question": "...", "answer": "..."}}"""

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"    Gemini error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="data/raw/candidates.jsonl")
    args = parser.parse_args()

    model = genai.GenerativeModel("gemini-1.5-flash")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_titles = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                if "nasa_title" in r:
                    existing_titles.add(r["nasa_title"].lower())

    generated = 0
    attempts = 0

    with out_path.open("a") as f:
        while generated < args.n and attempts < args.n * 5:
            attempts += 1
            target_date = random_date()

            try:
                apod = fetch_apod(target_date)
            except Exception as e:
                print(f"  NASA fetch error: {e}")
                time.sleep(2)
                continue

            if not apod:
                continue

            title = apod["title"]
            if title.lower() in existing_titles:
                continue
            existing_titles.add(title.lower())

            explanation = apod["explanation"]
            print(f"[{generated+1}/{args.n}] {title[:65]}")

            qa = generate_qa(model, title, explanation)
            if not qa or not qa.get("question") or not qa.get("answer"):
                continue

            record = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
                "reviewed": False,
                "source": "nasa_apod",
                "nasa_title": title,
                "nasa_date": str(target_date),
                "nasa_explanation": explanation,
            }
            f.write(json.dumps(record) + "\n")
            generated += 1
            time.sleep(0.3)

    print(f"\n{generated} NASA-grounded candidates appended to {out_path}")
    print("Review each one: compare the answer against nasa_explanation field,")
    print("set reviewed=true for accurate ones, then run prepare_dataset.py --force")


if __name__ == "__main__":
    main()
