"""Shared helpers for the v2 (audience-style) data pipeline."""
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))   # so `import prompting` works


class ModelUnavailable(Exception):
    """The configured Gemini model name is retired / not available to this key. Fix configs/data_pipeline.yaml."""


class QuotaExhausted(Exception):
    """Raised when the Gemini free-tier quota (usually the daily one) is used up. Re-run later; scripts resume."""


def load_cfg(path=None):
    p = Path(path) if path else ROOT / "configs" / "data_pipeline.yaml"
    return yaml.safe_load(p.read_text())


def load_styles():
    from prompting import load_styles as _ls
    return _ls(str(ROOT / "prompts" / "audience_styles.yaml"))


def shared_rules():
    """Voice/accuracy rules from the original Sagan template (everything before 'Question:')."""
    txt = (ROOT / "prompts" / "sagan_style_template.txt").read_text()
    return txt.split("Question:")[0].strip()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_jsonl(path, rec):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path, recs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gemini_client():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(ROOT / ".env")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def preflight(client, *models):
    """Fail fast (before any generation) if a configured model is not usable; list alternatives."""
    bad = []
    for m in dict.fromkeys(models):
        try:
            client.models.get(model=m)
        except Exception as e:  # noqa: BLE001
            bad.append((m, str(e)[:200]))
    if not bad:
        return
    print("Configured Gemini model(s) not available for your key:")
    for m, e in bad:
        print(f"  - {m}: {e}")
    try:
        names = sorted(x.name.replace("models/", "") for x in client.models.list()
                       if "generateContent" in (getattr(x, "supported_actions", None) or ["generateContent"]))
        print("\nModels your key can use (pick from these):")
        for n in names:
            if "gemini" in n:
                print("  ", n)
    except Exception as e:  # noqa: BLE001
        print(f"(could not list models: {e})")
    sys.exit("\nEdit generation.gen_model / filter.judge_model in configs/data_pipeline.yaml, then re-run.")


def parse_json(text):
    if not text:
        return None
    text = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def ask_json(client, model, prompt, temperature=0.7, retries=6):
    """Call Gemini expecting JSON. Backs off on rate limits; raises QuotaExhausted on daily quota."""
    from google.genai import types
    delay = 5
    bad_json = 0
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=temperature
                ),
            )
            return parse_json(r.text)
        except json.JSONDecodeError:
            bad_json += 1
            if bad_json >= 2:
                return None
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise QuotaExhausted(msg)
            if "404" in msg or "NOT_FOUND" in msg or "no longer available" in msg:
                raise ModelUnavailable(f"Model '{model}' is not available for this key: {msg[:300]}")
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE")):
                short = re.sub(r"\s+", " ", msg)[:200]
                print(f"  [{model}] rate-limited/unavailable; waiting {delay}s "
                      f"(attempt {attempt + 1}/{retries}): {short}", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            print(f"  API error: {msg[:200]}")
            return None
    raise QuotaExhausted("Retries exhausted (rate limit or outage). Re-run later; progress is saved.")


def word_count(s):
    return len(s.split())


def sentences(s):
    return [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
