#!/usr/bin/env python3

import argparse
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


async def dream(
    model: str,
    max_tokens: int,
    temperature: float,
    output_dir: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dream_t{temperature}_{timestamp}"

    tokens_file = output_dir / f"{run_name}_tokens.txt"
    meta_file = output_dir / f"{run_name}_meta.jsonl"
    raw_file = output_dir / f"{run_name}_raw.txt"

    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_dtype": kv_cache_dtype,
        "start_time": timestamp,
    }
    config_file = output_dir / f"{run_name}_config.json"
    config_file.write_text(json.dumps(config, indent=2))
    print(f"Config saved to {config_file}")

    print(f"Loading model {model}...")
    engine_args = AsyncEngineArgs(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=False,
        ignore_eos=True,
    )

    print(f"Starting generation...")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens:,}")
    print(f"  Output: {output_dir}")
    print("-" * 60)

    start_time = time.time()
    last_log_time = start_time
    last_token_count = 0
    total_tokens = 0

    with open(tokens_file, "w", encoding="utf-8") as f_tokens, \
         open(meta_file, "w", encoding="utf-8") as f_meta, \
         open(raw_file, "w", encoding="utf-8") as f_raw:

        prompt = " "

        async for output in engine.generate(prompt, params, request_id="dream"):
            generated = output.outputs[0].text
            new_tokens = len(output.outputs[0].token_ids)

            if new_tokens > total_tokens:
                current_text = generated

                f_raw.seek(0)
                f_raw.write(current_text)
                f_raw.flush()

                total_tokens = new_tokens

                now = time.time()
                if now - last_log_time >= 10:
                    elapsed = now - start_time
                    tokens_per_sec = (total_tokens - last_token_count) / (now - last_log_time)
                    overall_tps = total_tokens / elapsed

                    progress = {
                        "tokens": total_tokens,
                        "elapsed_sec": round(elapsed, 1),
                        "tokens_per_sec": round(tokens_per_sec, 1),
                        "overall_tps": round(overall_tps, 1),
                        "timestamp": time.time(),
                    }
                    f_meta.write(json.dumps(progress) + "\n")
                    f_meta.flush()

                    print(f"[{elapsed/60:.1f}m] {total_tokens:,} tokens | {tokens_per_sec:.1f} tok/s (avg: {overall_tps:.1f})")

                    last_log_time = now
                    last_token_count = total_tokens

        final_text = output.outputs[0].text
        f_raw.seek(0)
        f_raw.truncate()
        f_raw.write(final_text)

        f_tokens.write(final_text)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Done!")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Average speed: {total_tokens/elapsed:.1f} tokens/sec")
    print(f"  Output saved to: {output_dir}/{run_name}_*")

    summary = {
        **config,
        "end_time": datetime.now().isoformat(),
        "total_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "average_tokens_per_sec": total_tokens / elapsed,
    }
    summary_file = output_dir / f"{run_name}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=470_000,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=480_000,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
    )

    args = parser.parse_args()

    asyncio.run(dream(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
    ))


if __name__ == "__main__":
    main()
