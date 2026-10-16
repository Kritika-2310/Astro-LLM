"""
QLoRA fine-tuning using PEFT + TRL.
Run: python scripts/train.py --config configs/training_config.yaml
"""
import argparse
import inspect
import json
import math
import os
from collections import Counter
from pathlib import Path
 
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
 
from prompting import build_prompt, load_styles
 
load_dotenv()
 
 
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    run_name = cfg["run_name"]
    cfg["output"]["output_dir"] = cfg["output"]["output_dir"].replace("${run_name}", run_name)
    cfg["output"]["adapter_dir"] = cfg["output"]["adapter_dir"].replace("${run_name}", run_name)
    return cfg
 
 
def load_lora_config(path="configs/lora_config.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return LoraConfig(**raw)
 
 
def format_example(example, template, styles=None):
    question = example["instruction"]
    if example.get("input"):
        question = f"{question}\n{example['input']}"
    return build_prompt(template, question, example.get("style"), styles) + " " + example["output"]
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--override-rank", type=int, default=None)
    parser.add_argument("--override-lr", type=float, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    lora_cfg = load_lora_config()
 
    if args.override_rank:
        lora_cfg.r = args.override_rank
        lora_cfg.lora_alpha = args.override_rank * 2
    if args.override_lr:
        cfg["training"]["learning_rate"] = args.override_lr
    if args.override_epochs:
        cfg["training"]["num_train_epochs"] = args.override_epochs
 
    template = Path(cfg["data"]["prompt_template"]).read_text()
    os.environ.setdefault("WANDB_PROJECT", cfg["tracking"]["wandb_project"])
 
    # T4 (compute capability 7.5) has no native bf16 -> use fp16 there, bf16 on Ampere+
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")
 
    bnb_config = None
    if cfg["model"]["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=amp_dtype,
            bnb_4bit_use_double_quant=True,
        )
 
    print(f"Loading tokenizer: {cfg['model']['base_model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_id"],
        token=os.environ.get("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
 
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_id"],
        quantization_config=bnb_config,
        dtype=amp_dtype,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
 
    # Standard QLoRA prep: casts non-quantized params (norms, embeddings, lm_head) to fp32,
    # enables input grads and gradient checkpointing. Avoids bf16 params leaking into the
    # fp16 GradScaler on T4.
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
 
    model = get_peft_model(model, lora_cfg)
 
    # Safety net: GradScaler cannot unscale bf16 grads. Force every trainable param to fp32
    # when running in fp16 mode (LoRA weights stay fp32, autocast handles the fp16 math).
    if not use_bf16:
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float32)
 
    model.print_trainable_parameters()
    print("Trainable param dtypes:", dict(Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)))
 
    dataset = load_dataset("json", data_files={
        "train": cfg["data"]["train_path"],
        "validation": cfg["data"]["val_path"],
    })
 
    styles = load_styles(cfg["data"]["styles_path"]) if cfg["data"].get("styles_path") else None
 
    def _format(batch):
        n = len(batch["instruction"])
        sty = batch["style"] if "style" in batch else [None] * n   # v1 data has no style column
        return {"text": [
            format_example({"instruction": i, "input": inp, "output": o, "style": s}, template, styles)
            for i, inp, o, s in zip(batch["instruction"], batch["input"], batch["output"], sty)
        ]}
 
    dataset = dataset.map(_format, batched=True)
 
    t = cfg["training"]
    # transformers >= 5 removed `warmup_ratio` (use `warmup_steps`); older versions only know the ratio.
    if "warmup_ratio" in inspect.signature(SFTConfig.__init__).parameters:
        warmup = {"warmup_ratio": t["warmup_ratio"]}
    else:
        n_train = len(dataset["train"])
        eff_bs = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
        total_steps = math.ceil(n_train / eff_bs) * t["num_train_epochs"]
        warmup = {"warmup_steps": max(1, round(t["warmup_ratio"] * total_steps))}
        print(f"warmup_ratio unsupported by this transformers version -> warmup_steps={warmup['warmup_steps']} of {total_steps}")
    sft_config = SFTConfig(
        output_dir=cfg["output"]["output_dir"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        **warmup,
        lr_scheduler_type=t["lr_scheduler_type"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        report_to=cfg["tracking"]["report_to"],
        run_name=cfg["run_name"],
        seed=t["seed"],
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
    )
 
    # newer TRL names the argument `processing_class`; older versions call it `tokenizer`
    tok_kw = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=t["early_stopping_patience"]
        )],
        **{tok_kw: tokenizer},
    )
 
    # SFTTrainer init can re-cast adapter weights to bf16 behind our back. The optimizer is
    # created lazily inside train(), so re-cast to fp32 here, right before it is built.
    if not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
    print("Trainable param dtypes after trainer init:",
          dict(Counter(str(p.dtype) for p in trainer.model.parameters() if p.requires_grad)))
 
    print("Starting training...")
    trainer.train()
 
    adapter_dir = cfg["output"]["adapter_dir"]
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
 
    with open(Path(adapter_dir) / "run_manifest.json", "w") as f:
        json.dump({"config": cfg, "lora_config": lora_cfg.to_dict()}, f, indent=2, default=str)
 
    print(f"\nDone. Adapter saved to {adapter_dir}")
 
 
if __name__ == "__main__":
    main()
 