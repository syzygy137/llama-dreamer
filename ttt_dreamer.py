#!/usr/bin/env python3
"""
TTT Dreamer: Self-Training Collapse Dynamics

Model generates text continuously, trains on its own output via LoRA,
and we measure how/when it collapses.
"""

import argparse
import copy
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_distinct_n(token_ids, n):
    """Compute distinct-n: unique n-grams / total n-grams."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_weight_drift(model, initial_state):
    """L2 distance of LoRA weights from initialization."""
    total = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_state:
            diff = param.data.float() - initial_state[name].float()
            total += diff.pow(2).sum().item()
    return math.sqrt(total)


def run(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"ttt_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    config = vars(args)
    config["timestamp"] = timestamp
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Snapshot initial LoRA weights for drift measurement
    initial_lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Tokenize seed prompt
    context_tokens = tokenizer.encode(args.seed_prompt, return_tensors="pt").to(model.device)

    metrics_file = open(run_dir / "metrics.jsonl", "w")
    generated_file = open(run_dir / "generated.txt", "w")

    print(f"Output: {run_dir}")
    print(f"Steps: {args.num_steps}, LR: {args.lr}, Chunk: {args.chunk_size}")
    print(f"Control mode: {args.control}")
    print("-" * 60)

    start_time = time.time()

    for step in range(args.num_steps):
        step_start = time.time()

        # === GENERATE chunk (no grad) ===
        model.eval()
        with torch.no_grad():
            gen_output = model.generate(
                context_tokens,
                max_new_tokens=args.chunk_size,
                temperature=args.temperature,
                do_sample=True,
                top_k=0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_token_ids = gen_output.sequences[0, context_tokens.shape[1]:]
        new_tokens_list = new_token_ids.tolist()
        text = tokenizer.decode(new_token_ids, skip_special_tokens=False)

        # Token entropy and top1 prob from generation scores
        token_entropies = []
        top1_probs = []
        for score in gen_output.scores:
            probs = torch.softmax(score[0].float(), dim=-1)
            log_probs = torch.log_softmax(score[0].float(), dim=-1)
            entropy = -(probs * log_probs).nan_to_num(0.0).sum().item()
            token_entropies.append(entropy)
            top1_probs.append(probs.max().item())

        avg_token_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
        avg_top1_prob = sum(top1_probs) / len(top1_probs) if top1_probs else 0.0

        # === TRAIN on generated chunk ===
        grad_norm_val = 0.0
        if not args.control:
            model.train()
            input_ids = torch.cat([context_tokens, new_token_ids.unsqueeze(0)], dim=1)

            # Truncate to max_context if needed
            if input_ids.shape[1] > args.max_context:
                input_ids = input_ids[:, -args.max_context:]

            labels = input_ids.clone()
            # Mask context tokens so loss only applies to generated tokens
            ctx_len = min(context_tokens.shape[1], input_ids.shape[1])
            if input_ids.shape[1] == context_tokens.shape[1] + len(new_tokens_list):
                # No truncation happened
                labels[:, :context_tokens.shape[1]] = -100
            else:
                # Truncation happened; mask everything except the new tokens at the end
                labels[:, :-len(new_tokens_list)] = -100

            output = model(input_ids=input_ids, labels=labels)
            loss_val = output.loss.item()

            train_loss = -output.loss if args.negative else output.loss
            train_loss.backward()

            # Compute grad norm
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm += p.grad.data.float().pow(2).sum().item()
            grad_norm_val = math.sqrt(total_norm)

            optimizer.step()
            optimizer.zero_grad()
        else:
            # Control: still compute loss for logging but don't train
            model.eval()
            with torch.no_grad():
                input_ids = torch.cat([context_tokens, new_token_ids.unsqueeze(0)], dim=1)
                if input_ids.shape[1] > args.max_context:
                    input_ids = input_ids[:, -args.max_context:]
                labels = input_ids.clone()
                if input_ids.shape[1] == context_tokens.shape[1] + len(new_tokens_list):
                    labels[:, :context_tokens.shape[1]] = -100
                else:
                    labels[:, :-len(new_tokens_list)] = -100
                output = model(input_ids=input_ids, labels=labels)
                loss_val = output.loss.item()

        # === SLIDE context window ===
        context_tokens = torch.cat([context_tokens, new_token_ids.unsqueeze(0)], dim=1)
        if context_tokens.shape[1] > args.max_context:
            context_tokens = context_tokens[:, -args.max_context:]

        # === METRICS ===
        perplexity = math.exp(loss_val) if loss_val < 20 else float("inf")
        distinct_1 = compute_distinct_n(new_tokens_list, 1)
        distinct_2 = compute_distinct_n(new_tokens_list, 2)
        weight_drift = compute_weight_drift(model, initial_lora_state) if not args.control else 0.0

        metrics = {
            "step": step,
            "loss": round(loss_val, 6),
            "perplexity": round(perplexity, 4) if perplexity != float("inf") else "inf",
            "distinct_1": round(distinct_1, 6),
            "distinct_2": round(distinct_2, 6),
            "token_entropy": round(avg_token_entropy, 6),
            "top1_prob": round(avg_top1_prob, 6),
            "weight_drift": round(weight_drift, 6),
            "grad_norm": round(grad_norm_val, 6),
            "text_snippet": text[:200],
            "elapsed": round(time.time() - start_time, 1),
            "step_time": round(time.time() - step_start, 2),
        }

        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

        generated_file.write(text)
        generated_file.flush()

        # Checkpoint
        if step % args.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:04d}"
            model.save_pretrained(ckpt_path)
            print(f"  [checkpoint saved: {ckpt_path}]")

        # Log
        print(
            f"[step {step:4d}] loss={loss_val:.4f} ppl={perplexity:8.2f} "
            f"d1={distinct_1:.3f} d2={distinct_2:.3f} "
            f"ent={avg_token_entropy:.2f} top1={avg_top1_prob:.3f} "
            f"drift={weight_drift:.4f} gnorm={grad_norm_val:.4f} "
            f"| {text[:80]!r}"
        )

    # === SUMMARY ===
    elapsed = time.time() - start_time
    summary = {
        **config,
        "end_time": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "total_steps": args.num_steps,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    metrics_file.close()
    generated_file.close()

    print("-" * 60)
    print(f"Done! {args.num_steps} steps in {elapsed / 60:.1f} minutes")
    print(f"Output: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="TTT Dreamer: Self-Training Collapse Dynamics")
    parser.add_argument("--model", default="gradientai/Llama-3-8B-Instruct-Gradient-1048k")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--seed-prompt", default=" ")
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--control", action="store_true", help="Skip training step (baseline comparison)")
    parser.add_argument("--negative", action="store_true", help="Negate loss (train away from own output)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
