"""
Test Cheers text generation via vLLM's native CheersForConditionalGeneration.

This uses vLLM's LLM engine to run the Cheers model for pure text (chat)
generation, demonstrating that vllm-omni can leverage vLLM's text path.

Usage:
  python test_vllm_text_gen.py --model /path/to/Cheers
"""

import argparse
import time

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Cheers vLLM text generation")
    parser.add_argument("--model", default="/home/liubingshuai/data1/models/Cheers")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def build_chat_prompt(user_msg: str) -> str:
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    args = parse_args()

    print(f"Loading Cheers model from {args.model} via vLLM engine...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    test_prompts = [
        "What is the capital of France? Answer briefly.",
        "Explain what a neural network is in 2-3 sentences.",
        "Write a short haiku about spring.",
        "What are the main differences between Python and C++?",
        "Summarize the concept of attention mechanism in transformers.",
    ]

    prompts = [build_chat_prompt(p) for p in test_prompts]

    print(f"\nGenerating {len(prompts)} responses...\n")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1

    total_tokens = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        n_tok = len(output.outputs[0].token_ids)
        total_tokens += n_tok
        print(f"{'='*60}")
        print(f"[Prompt {i+1}] {test_prompts[i]}")
        print(f"[Output] {text}")
        print(f"[Tokens] {n_tok}")

    print(f"\n{'='*60}")
    print(f"Total generation time: {gen_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {total_tokens / gen_time:.1f} tokens/s")
    print(f"Average latency: {gen_time / len(prompts):.2f}s per prompt")


if __name__ == "__main__":
    main()
