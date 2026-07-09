import argparse
import json
import os
import re
import sys
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


def log(msg):
    print(msg, flush=True)


# 当 url(#id) 引用的渐变被剥离后，用来顶替的纯色回退调色板。
_FALLBACK_PALETTE = ["#3B6EA5", "#D9D9D9", "#2B2B2B", "#E08A1E", "#6FAF3C"]


def flatten_svg(svg_text: str) -> str:
    """把训练目标 SVG 拍平成「形状 + 纯色」结构。

    Sonnet 生成的训练 SVG 大量使用 ``<defs>`` 内的 ``<linearGradient>`` /
    ``<radialGradient>`` 与 ``<stop>`` 子元素。270M 的 LoRA 模型无法可靠地
    关闭这种深层嵌套——它会自闭合 ``<linearGradient .../>`` 再追加子元素，
    或干脆忘了 ``</defs>``，导致整个文档不可解析。

    本函数彻底移除 ``<defs>``，并把每个 ``fill="url(#id)"`` /
    ``stroke="url(#id)"`` 替换成从对应渐变首个 ``<stop>`` 取到的纯色
    （取不到则用调色板回退）。结果是一个结构简单、仍能代表同一徽标的
    SVG，模型更容易学会正确闭合。
    """
    # 从每个渐变里 harvest 首个 stop-color，按 id 索引。
    grad_map: Dict[str, str] = {}
    for m in re.finditer(
        r'<(?:linear|radial)Gradient\b[^>]*\bid=["\']([^"\']+)["\'][^>]*>(.*?)</(?:linear|radial)Gradient\s*>',
        svg_text, flags=re.DOTALL,
    ):
        gid, body = m.group(1), m.group(2)
        stop_m = re.search(r'stop-color=["\']([^"\']+)["\']', body)
        if stop_m:
            grad_map[gid] = stop_m.group(1)

    # 自闭合渐变：<linearGradient id="x" ... stop-color="#.."/>
    for m in re.finditer(
        r'<(?:linear|radial)Gradient\b[^>]*\bid=["\']([^"\']+)["\'][^>]*/>',
        svg_text,
    ):
        gid = m.group(1)
        sc = re.search(r'stop-color=["\']([^"\']+)["\']', m.group(0))
        if sc:
            grad_map[gid] = sc.group(1)

    out = svg_text
    # 删除所有 <defs>...</defs> 块（贪婪匹配）。
    out = re.sub(r'<defs\b[^>]*>.*?</defs\s*>', '', out, flags=re.DOTALL)
    # 删除无闭合标签的孤儿 <defs ...>。
    out = re.sub(r'<defs\b[^>]*>(?:(?!</svg>).)*$', '', out, flags=re.DOTALL)

    # 把 url(#id) 引用替换成 harvest 到的纯色。
    def _swap_url(m):
        gid = m.group(1)
        return grad_map.get(gid, _FALLBACK_PALETTE[abs(hash(gid)) % len(_FALLBACK_PALETTE)])

    out = re.sub(r'url\(#([^)]+)\)', _swap_url, out)
    return out


class LogoDataset(Dataset):
    """SVG 徽标 LoRA 训练数据集。

    几个不可妥协的设计点：
    - prompt 用 ``add_generation_prompt=True`` 拼装，结尾带 assistant 轮标记，
      和 eval 时的生成前缀完全一致（否则训练/eval 分布不匹配）。
    - assistant（SVG）内容**绝不截断**：放不进 ``max_length`` 的样本直接跳过。
      截断会从文档中间砍掉 ``</svg>``，教模型输出不闭合的文档——这是早期
      配置导致微调回归的主因。
    - SVG 末尾追加 EOS，让模型学会在 ``</svg>`` 后停止生成。
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        skipped = 0

        eos_id = tokenizer.eos_token_id

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                messages = item["messages"]

                prompt_msgs = [m for m in messages if m["role"] != "assistant"]
                assistant_msgs = [m for m in messages if m["role"] == "assistant"]
                if not assistant_msgs:
                    skipped += 1
                    continue
                svg_target = assistant_msgs[0]["content"]
                # 拍平 SVG 目标：剥离 <defs>/渐变，模型才能学会闭合。
                svg_target = flatten_svg(svg_target)

                # 训练 prompt 与 eval 生成前缀保持一致。
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = self.tokenizer(
                    prompt_text, padding=False, return_tensors=None
                )["input_ids"]

                target_ids = self.tokenizer(
                    svg_target,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )["input_ids"]
                # 教模型在 </svg> 后停止。
                target_ids = target_ids + [eos_id]

                input_ids = prompt_ids + target_ids
                if len(input_ids) > self.max_length:
                    # 跳过而非截断：截断的 SVG 缺 </svg>，会教模型产出不可闭合输出。
                    skipped += 1
                    continue

                self.records.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": [-100] * len(prompt_ids) + target_ids,
                    }
                )

        log(f"Loaded {len(self.records)} samples from {jsonl_path} (skipped {skipped} over-length/empty).")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return {k: list(v) for k, v in self.records[idx].items()}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SVG logo generation")
    parser.add_argument("--train_data", type=str, default="../logo-detailed-prompt/train.jsonl")
    parser.add_argument("--valid_data", type=str, default="../logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="../models/google/gemma-3-270m-it")
    parser.add_argument("--output_dir", type=str, default="../adapter")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop if eval_loss does not improve for N evals")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_dataset = LogoDataset(args.train_data, tokenizer, args.max_length)
    valid_dataset = LogoDataset(args.valid_data, tokenizer, args.max_length)

    log(f"\n=== Dataset Sanity Check ===")
    log(f"Train samples: {len(train_dataset)}")
    log(f"Valid samples: {len(valid_dataset)}")
    sample = train_dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    valid_labels = [l for l in labels if l != -100]
    log(f"Sample 0 - input_ids len: {len(input_ids)}, labels len: {len(labels)}")
    log(f"Sample 0 - valid (non -100) labels: {len(valid_labels)}")
    log(f"Sample 0 - first 10 valid labels: {valid_labels[:10]}")
    log(f"Sample 0 - decoded prompt tail: {tokenizer.decode(input_ids[:50])[-100:]}")
    log(f"Sample 0 - decoded assistant head: {tokenizer.decode(valid_labels[:20])}")
    if len(valid_labels) == 0:
        log("ERROR: No valid labels found! Training will produce 0 loss.")
        sys.exit(1)

    log(f"\n=== Forward Pass Test ===")
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([sample["input_ids"]]).to(model.device)
        test_labels = torch.tensor([sample["labels"]]).to(model.device)
        test_attn = torch.tensor([sample["attention_mask"]]).to(model.device)
        outputs = model(input_ids=test_input, attention_mask=test_attn, labels=test_labels)
        log(f"Test loss: {outputs.loss.item()}")
        log(f"Logits shape: {outputs.logits.shape}")
        if torch.isnan(outputs.loss):
            log("ERROR: Loss is NaN even in forward pass!")
            sys.exit(1)
        log(f"=== Forward Pass OK ===\n")
    model.train()

    log(f"=== Sanity Check Passed ===\n")

    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = tokenizer.pad_token_id
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}

    data_collator = collator

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience,
                                         early_stopping_threshold=0.001)],
    )

    log("Starting training...")
    trainer.train()

    log("Saving adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    log(f"Training complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
