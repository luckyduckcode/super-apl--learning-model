"""
LoRA training script (scaffold)

This script expects a `dataset.jsonl` where each line is a JSON object with keys: {"instruction": ..., "output": ...}.
It trains a LoRA adapter using PEFT & bitsandbytes (4-bit) and saves the adapter in `./adapters/`.

Note: This is a skeleton and requires `transformers`, `datasets`, `peft`, `bitsandbytes`, and `accelerate`.
"""

import os
import json

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling
    BITSANDBYTES_AVAILABLE = True
except Exception:
    BITSANDBYTES_AVAILABLE = False


def train_lora(model_name_or_path: str, dataset_path: str, output_dir: str = './adapters/mylora', lr=2e-4, per_device_train_batch_size=4, epochs=3, load_in_4bit: bool = True):
    if not BITSANDBYTES_AVAILABLE:
        raise RuntimeError("Required modules not installed: transformers, datasets, peft, bitsandbytes, accelerate")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    dataset = load_dataset('json', data_files={'train': dataset_path})['train']

    def preprocess(example):
        # For CAUSAL LM training, concatenate instruction and output
        text = example.get('instruction', '') + "\n\n" + example.get('output', '')
        return tokenizer(text, truncation=True, max_length=512)

    tokenized = dataset.map(preprocess, batched=False)

    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = prepare_model_for_kbit_training(model)

    # Auto-detect target modules for LoRA depending on model architecture
    # Common choices:
    # - Transformers with separate Q/K/V/O: q_proj, k_proj, v_proj, o_proj
    # - GPT-2 style Conv1D attention: c_attn, c_proj
    model_module_names = [n for n, _ in model.named_modules()]
    if any('q_proj' in name for name in model_module_names):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif any('c_attn' in name for name in model_module_names):
        target_modules = ["c_attn", "c_proj"]
    else:
        # fallback; attempt both lists
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)

    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--dataset', default='./training/library_dataset.jsonl')
    parser.add_argument('--output', default='./adapters/mylora')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit loading for testing on small models')
    args = parser.parse_args()

    train_lora(args.model, args.dataset, args.output, lr=args.lr, per_device_train_batch_size=args.batch, epochs=args.epochs, load_in_4bit=not args.no_4bit)
