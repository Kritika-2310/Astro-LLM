# Astro-LLM — Sagan-Style Astronomy Explainer

Fine-tuned Llama 3.1 8B to answer astronomy questions with the vivid, scientifically rigorous voice of Carl Sagan and Neil deGrasse Tyson.

**Base model:** dry, textbook answers  
**Fine-tuned:** warm, analogy-driven explanations that never sacrifice accuracy

---

## Before / After

**Question:** What is a black hole?

**Base Llama 3.1 8B:**
> A black hole is a region in space where the gravitational pull is so strong that nothing, not even light, can escape. It forms when a massive star collapses at the end of its life...

**Astro-LLM (fine-tuned):**
> Imagine spacetime as a cosmic fabric, stretched taut across the universe. A black hole isn't just a void — it's where gravity curves that fabric so intensely that spacetime folds in on itself, creating a point of no return called the event horizon. Even light, the fastest thing in the universe, cannot climb back out once it crosses that threshold. This isn't science fiction; it's the elegant, terrifying consequence of Einstein's general relativity playing out in the most extreme environments the cosmos can produce.

---

## Results

| Metric | Value |
|--------|-------|
| Training loss | 1.68 → 0.45 |
| Eval loss | 0.494 |
| Token accuracy | 86.35% |
| Training time | ~70 mins on T4 |
| Trainable params | 41.9M / 8.07B (0.52%) |

*Evaluation scores (base vs fine-tuned on 15-question benchmark) — coming soon*

---

## Architecture

```
Base model:   meta-llama/Meta-Llama-3.1-8B-Instruct
Method:       QLoRA (4-bit NF4) + LoRA rank 16
Target:       q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Alpha:        32  |  Dropout: 0.05
Epochs:       3   |  LR: 2e-4 (cosine schedule)
Batch size:   2 × 8 gradient accumulation = effective 16
```

---

## Dataset

- 159 unique astronomy Q&A pairs
- Generated with Groq (Llama 3.3 70B) using a Sagan-style prompt specification
- Every example manually reviewed and fact-checked against NASA and OpenStax Astronomy
- Split: 127 train / 15 val / 17 test
- Eval benchmark: 15 handcrafted questions with required facts and known pitfalls per question

---

## Repo Structure

```
Astro-LLM/
├── configs/
│   ├── lora_config.yaml        # LoRA hyperparameters
│   └── training_config.yaml    # Full training config + sweep grid
├── data/
│   ├── processed/              # train / val / test splits
│   ├── raw/topics.txt          # 250 seed questions
│   └── eval_benchmark.jsonl   # Handcrafted evaluation set
├── prompts/
│   └── sagan_style_template.txt  # Style specification
├── scripts/
│   ├── generate_dataset.py    # Groq-powered candidate generation
│   ├── prepare_dataset.py     # Dedup, leakage check, 80/10/10 split
│   ├── train.py               # QLoRA training with W&B logging
│   ├── evaluate.py            # Rule-based + Gemini-as-judge scoring
│   └── compare_results.py     # Base vs fine-tuned head-to-head
├── serve/
│   └── inference_server.py    # FastAPI A/B comparison endpoint
└── outputs/
    └── astro-llm-r16-lr2e4-e3/
        └── final_adapter/     # Adapter config (weights on HuggingFace)
```

---

## Adapter Weights

Stored on HuggingFace (160MB safetensors, too large for GitHub):

```
huggingface.co/kritika2322/astro-llm-llama3-lora
```

Load and run:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, "kritika2322/astro-llm-llama3-lora")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
```

---

## Reproduce Training

```bash
git clone https://github.com/Kritika-2310/Astro-LLM
cd Astro-LLM
cp .env.example .env  # fill in GROQ_API_KEY, GEMINI_API_KEY, HF_TOKEN, WANDB_API_KEY

# Dataset generation (CPU, ~5 mins)
pip install -r requirements-dev.txt
python scripts/generate_dataset.py --n 200
# review data/raw/candidates.jsonl, set reviewed=true, then:
python scripts/prepare_dataset.py

# Training (requires GPU)
pip install -r requirements.txt
python scripts/train.py --config configs/training_config.yaml

# Evaluate
python scripts/evaluate.py --model-path base --label base_model
python scripts/evaluate.py --model-path outputs/astro-llm-r16-lr2e4-e3/final_adapter --label finetuned
python scripts/compare_results.py
```

---

## Experiment Tracking

W&B run: `kritika23ojha-mangalmay-group-of-institutions/astro-llm`

Training loss curve shows steady descent from 1.68 → 0.45 over 3 epochs with no sign of overfitting (eval loss 0.494 tracks closely).

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Llama 3.1 8B Instruct |
| Fine-tuning | PEFT + TRL (QLoRA) |
| Data generation | Groq API (Llama 3.3 70B) |
| LLM judge | Gemini 1.5 Flash |
| Experiment tracking | Weights & Biases |
| Inference | FastAPI + PEFT adapter loading |
| GPU | Google Colab T4 (free tier) |

---

## What This Demonstrates

- End-to-end ownership of the fine-tuning lifecycle: data curation → training → evaluation → serving
- QLoRA on consumer GPU hardware (T4, 15GB VRAM)
- Rigorous evaluation: rule-based fact coverage + LLM-as-judge, not just perplexity
- Reproducible pipeline — every run is config-driven and W&B logged
- Style transfer as a measurable ML task, not just vibe-checking

---

*Evaluation scores will be updated once the base vs fine-tuned benchmark run completes.*
