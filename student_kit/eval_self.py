import argparse
import gc
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from reward import compute_reward


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def load_model(model_path: str, adapter_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if adapter_path is not None and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def generate_svg(model, tokenizer, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.2) -> str:
    """用给定模型生成单个 SVG。

    生成参数选择理由（来自实测 smoke test）：
    - temperature=0.2 低温：偏好多走已学到的 SVG 语法，减少随机退化循环。
      smoke test 证实 temp=0.3 会让模型陷入 "L 18 L 18 L 18..." 重复死循环，
      撞满 1024 tokens 才停（45s/样本），整段垃圾 reward=0。
    - repetition_penalty=1.05（不能更高）：SVG 合法地大量重复（几十个 <circle>、
      反复的 fill=/stroke=/L 命令）。penalty=1.15 会逼模型生造垃圾 token 规避
      重复，report.md §6.3 已记录此教训。
    - 不用 no_repeat_ngram_size：会禁掉 SVG 必需的 3-gram 重复（如多个 <circle）。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a vector-logo designer. Read the description and emit exactly ONE SVG document.\n"
                "Hard rules:\n"
                "- Output only the SVG element: <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 256 256\">...</svg>.\n"
                "  No prose, no markdown, no code fences.\n"
                "- Keep all content centered inside roughly 20..236 on both axes.\n"
                "- Use 2 to 12 distinct colors; keep the palette cohesive.\n"
                "- Only vector primitives allowed: path, circle, ellipse, rect, polygon, polyline, line, g.\n"
                "  Do NOT use image, script, iframe, foreignObject, or any external URL/href.\n"
                "- Faithfully draw what the description specifies."
            ),
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return extract_svg(generated_text)


def _is_degenerate(text: str) -> bool:
    """检测模型输出是否陷入退化重复循环。

    270M 小模型在 do_sample 时，某些 prompt 会触发病态重复，典型表现：
    - 同一个 <circle .../> 完全重复几十次
    - path d="M128 128 L128 128 L128 128..." 坐标死循环

    检测策略：找 8 字符以上的子串，若重复出现 5 次以上，判定退化。
    合法 SVG 也会重复相似元素，但坐标/颜色不同——完全相同的子串重复
    5 次几乎只出现在退化输出里。
    """
    import re
    # 检测 8+ 字符子串重复 5+ 次
    if re.search(r'(.{8,}?)\1{4,}', text):
        return True
    # 检测同一个元素标签重复 5+ 次（如 <circle .../> 连续相同）
    if re.search(r'(<\w+[^>]*?/>\s*)\1{4,}', text):
        return True
    return False


def extract_svg(text: str) -> str:
    """从模型原始输出里稳健地抠出单个 <svg>...</svg> 文档。

    小模型最常见的失败模式：输出了开标签 <svg> 但永远不输出闭标签
    </svg>（通常是撞到 token 预算）。此时追加一个闭标签，让文档至少能被
    解析，而不是整段被判 0 分。
    """
    text = text.strip()

    # 剥掉 markdown 代码围栏
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()

    svg_start = text.find("<svg")
    if svg_start == -1:
        # 完全没有 svg 开标签，原样返回让 reward 报告无效
        return text

    svg_end = text.rfind("</svg>")
    if svg_end != -1 and svg_end > svg_start:
        return text[svg_start:svg_end + 6]

    # 有开标签但没闭标签：截到最后一个 '>'，再补 </svg>
    fragment = text[svg_start:]
    last_gt = fragment.rfind(">")
    if last_gt != -1 and last_gt < len(fragment) - 1:
        fragment = fragment[:last_gt + 1]
    return fragment + "</svg>"


def evaluate_model(model, tokenizer, samples, label, max_new_tokens=1024, temperature=0.2, num_samples=2):
    """对每个 prompt 采样多次取 reward 最高的，并对退化输出自动重采样。

    270M 小模型用 do_sample=True 时，某些 seed 会触发病态重复循环
    （如 "L128 128 L128 128 L128 128..." 或完全相同的 <circle> 重复数十次）。
    单次生成可能整段退化 reward=0。两层对策：
    1. 生成后用 _is_degenerate 检测退化——检测到就立即丢弃重采样，
       不浪费 reward 计算，最多额外重试 num_samples 次。
    2. 未退化但 reward 仍低的样本，保留备选，最终取 reward 最高。
    """
    results = []
    for i, sample in enumerate(samples):
        log(f"  [{label}] Sample {i+1}/{len(samples)} (n={num_samples})...")

        prompt = None
        for msg in sample["messages"]:
            if msg["role"] == "user":
                prompt = msg["content"]

        if prompt is None:
            continue

        best = None
        for k in range(num_samples):
            svg = generate_svg(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            if _is_degenerate(svg):
                log(f"      sub {k+1}/{num_samples}: DEGENERATE (重复循环), 跳过")
                continue
            reward = compute_reward(svg, prompt)
            if best is None or reward["total"] > best["reward"]["total"]:
                best = {"svg": svg, "reward": reward, "sample_idx": k}
            log(f"      sub {k+1}/{num_samples}: total={reward['total']} valid={reward['valid']}")

        # 如果所有采样都退化（best is None），额外重试一次，放宽退化阈值
        if best is None:
            log(f"      全部退化, 额外重采样 (retry)...")
            for _ in range(num_samples):
                svg = generate_svg(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                reward = compute_reward(svg, prompt)
                best = {"svg": svg, "reward": reward, "sample_idx": -1}
                if reward["total"] > 0:
                    break
            log(f"      retry: total={best['reward']['total']} valid={best['reward']['valid']}")

        results.append({
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "svg": best["svg"],
            "reward": best["reward"]
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned model on SVG generation")
    parser.add_argument("--valid_data", type=str, default="../logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="../models/google/gemma-3-270m-it")
    parser.add_argument("--adapter_path", type=str, default="../adapter")
    parser.add_argument("--output_file", type=str, default="../results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_samples", type=int, default=2,
                        help="每个 prompt 采样次数，取 reward 最高（对抗 do_sample 退化循环）")
    args = parser.parse_args()

    log("Loading validation data...")
    valid_samples = []
    with open(args.valid_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                valid_samples.append(json.loads(line))

    if args.max_samples is not None:
        valid_samples = valid_samples[:args.max_samples]

    log(f"Total samples: {len(valid_samples)}")

    # Phase 1: 基座评测
    log("\n=== Phase 1: Baseline Evaluation ===")
    log("Loading baseline model...")
    baseline_model, tokenizer = load_model(args.model_name_or_path)

    baseline_results = evaluate_model(baseline_model, tokenizer, valid_samples, "baseline", args.max_new_tokens, args.temperature, args.num_samples)

    log("Freeing baseline model...")
    free_model(baseline_model)

    # Phase 2: 微调评测
    log("\n=== Phase 2: Fine-tuned Evaluation ===")
    log("Loading fine-tuned model...")
    ft_model, tokenizer = load_model(args.model_name_or_path, args.adapter_path)

    ft_results = evaluate_model(ft_model, tokenizer, valid_samples, "fine-tuned", args.max_new_tokens, args.temperature, args.num_samples)

    log("Freeing fine-tuned model...")
    free_model(ft_model)

    # 汇总
    baseline_scores = [r["reward"]["total"] for r in baseline_results]
    ft_scores = [r["reward"]["total"] for r in ft_results]

    baseline_valid_count = sum(1 for r in baseline_results if r["reward"]["valid"])
    ft_valid_count = sum(1 for r in ft_results if r["reward"]["valid"])

    results = {
        "baseline": baseline_results,
        "fine_tuned": ft_results,
        "summary": {
            "num_samples": len(valid_samples),
            "baseline": {
                "avg_reward": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0,
                "min_reward": min(baseline_scores) if baseline_scores else 0,
                "max_reward": max(baseline_scores) if baseline_scores else 0,
                "valid_count": baseline_valid_count,
                "valid_ratio": baseline_valid_count / len(baseline_scores) if baseline_scores else 0
            },
            "fine_tuned": {
                "avg_reward": sum(ft_scores) / len(ft_scores) if ft_scores else 0,
                "min_reward": min(ft_scores) if ft_scores else 0,
                "max_reward": max(ft_scores) if ft_scores else 0,
                "valid_count": ft_valid_count,
                "valid_ratio": ft_valid_count / len(ft_scores) if ft_scores else 0
            },
            "delta": {
                "avg_reward": (sum(ft_scores) / len(ft_scores) if ft_scores else 0) - (sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0),
                "valid_count": ft_valid_count - baseline_valid_count
            }
        }
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved to {args.output_file}")
    log("\n=== Summary ===")
    log(f"Baseline avg reward: {results['summary']['baseline']['avg_reward']:.4f}")
    log(f"Fine-tuned avg reward: {results['summary']['fine_tuned']['avg_reward']:.4f}")
    log(f"Delta: {results['summary']['delta']['avg_reward']:.4f}")
    log(f"Baseline valid: {baseline_valid_count}/{len(valid_samples)}")
    log(f"Fine-tuned valid: {ft_valid_count}/{len(valid_samples)}")


if __name__ == "__main__":
    main()
