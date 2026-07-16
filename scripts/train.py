"""
QLoRA fine-tuning using PEFT + TRL.
Run: python scripts/train.py --config configs/training_config.yaml
"""
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

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


def format_example(example, template):
    question = example["instruction"]
    if example.get("input"):
        question = f"{question}\n{example['input']}"
    return template.format(question=question) + " " + example["output"]


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

    bnb_config = None
    if cfg["model"]["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": cfg["data"]["train_path"],
        "validation": cfg["data"]["val_path"],
    })

    def _format(batch):
        return {"text": [
            format_example({"instruction": i, "input": inp, "output": o}, template)
            for i, inp, o in zip(batch["instruction"], batch["input"], batch["output"])
        ]}

    dataset = dataset.map(_format, batched=True)

    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=cfg["output"]["output_dir"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
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
        bf16=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=t["early_stopping_patience"]
        )],
    )

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
