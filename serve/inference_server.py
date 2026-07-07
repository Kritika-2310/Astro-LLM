"""
FastAPI A/B comparison server.
POST /compare  ->  {"question": "..."} returns base + finetuned answers side by side.
POST /generate/base
POST /generate/finetuned
GET  /health

Run: uvicorn serve.inference_server:app --host 0.0.0.0 --port 8000
"""
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "outputs/final_adapter")
TEMPLATE = Path("prompts/sagan_style_template.txt").read_text()

app = FastAPI(title="Sagan Astro LLM")
tokenizer = base_model = finetuned_model = None

class Query(BaseModel):
    question: str
    max_new_tokens: int = 300

@app.on_event("startup")
def load_models():
    global tokenizer, base_model, finetuned_model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=os.environ.get("HF_TOKEN"))
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    )
    if Path(ADAPTER_PATH).exists():
        finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Adapter loaded from {ADAPTER_PATH}")
    else:
        print(f"No adapter at {ADAPTER_PATH} — finetuned endpoints will return a warning")

def _gen(model, question, max_new_tokens):
    prompt = TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

@app.post("/generate/base")
def gen_base(q: Query):
    return {"model": "base", "answer": _gen(base_model, q.question, q.max_new_tokens)}

@app.post("/generate/finetuned")
def gen_finetuned(q: Query):
    if not finetuned_model:
        return {"error": "No adapter loaded. Run training first."}
    return {"model": "finetuned", "answer": _gen(finetuned_model, q.question, q.max_new_tokens)}

@app.post("/compare")
def compare(q: Query):
    base_ans = _gen(base_model, q.question, q.max_new_tokens)
    ft_ans = _gen(finetuned_model, q.question, q.max_new_tokens) if finetuned_model else "No adapter loaded."
    return {"question": q.question, "base": base_ans, "finetuned": ft_ans}

@app.get("/health")
def health():
    return {"status": "ok", "adapter_loaded": finetuned_model is not None}
