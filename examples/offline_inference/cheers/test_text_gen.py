"""
Cheers autoregressive text generation test.

Tests the LLM (understanding/text) part of the Cheers model
using the standalone UMMTextModel from cheers_modules.

Usage:
  python test_text_gen.py --model /path/to/Cheers
"""

import argparse
import glob
import json
import os
import time

import torch
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Cheers text generation test")
    parser.add_argument(
        "--model",
        default="ai9stars/Cheers",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_language_model(model_path: str, device: torch.device):
    """Load the UMMTextModel with pretrained weights."""
    from vllm_omni.diffusion.models.cheers.cheers_modules import (
        CheersQwen2Config,
        UMMTextModel,
    )
    from safetensors import safe_open

    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)

    text_config = CheersQwen2Config.from_dict(raw.get("text_config", {}))
    model = UMMTextModel(text_config)

    lm_head_weight = None
    state = model.state_dict()
    allowed = set(state.keys())
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    loaded_count = 0

    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if key == "model.language_model.lm_head.weight" or key == "lm_head.weight":
                    lm_head_weight = tensor
                    continue

                name = key
                for pfx_src, pfx_dst in [
                    ("model.language_model.", ""),
                    ("language_model.", ""),
                ]:
                    if name.startswith(pfx_src):
                        name = pfx_dst + name[len(pfx_src):]
                        break

                if name in allowed and tuple(tensor.shape) == shapes.get(name):
                    param = model
                    parts = name.split(".")
                    for part in parts[:-1]:
                        param = getattr(param, part)
                    target = getattr(param, parts[-1])
                    if isinstance(target, torch.nn.Parameter):
                        target.data.copy_(tensor)
                    else:
                        target.copy_(tensor)
                    loaded_count += 1

    print(f"[Info] Loaded {loaded_count} language model weights")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, lm_head_weight, text_config


@torch.no_grad()
def generate_text(
    model,
    lm_head_weight: torch.Tensor,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    device: torch.device = torch.device("cuda"),
):
    """Simple autoregressive text generation using UMMTextModel."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    inputs_embeds = model.embed_tokens(input_ids)
    seq_len = inputs_embeds.size(1)
    head_num = model.config.num_attention_heads

    causal_mask = torch.tril(
        torch.ones((1, head_num, seq_len, seq_len), dtype=torch.long, device=device)
    )

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=causal_mask,
        use_cache=True,
        output_hidden_states=True,
    )
    past_kv = outputs.past_key_values

    lm_head = lm_head_weight.to(device=device, dtype=torch.bfloat16)
    hidden = outputs.last_hidden_state[:, -1:, :]
    logits = torch.matmul(hidden, lm_head.T)

    eos_token_id = tokenizer.eos_token_id or 151645
    generated_ids = []

    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                idx_in_topk = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, idx_in_topk)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id == eos_token_id:
            break

        generated_ids.append(token_id)
        next_embeds = model.embed_tokens(next_token)

        cur_len = past_kv.get_seq_length() + 1
        step_mask = torch.ones((1, head_num, 1, cur_len), dtype=torch.long, device=device)

        outputs = model(
            inputs_embeds=next_embeds,
            attention_mask=step_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1:, :]
        logits = torch.matmul(hidden, lm_head.T)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] Loading model from {args.model} ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, lm_head_weight, text_config = load_language_model(args.model, device)

    print(f"[Info] Model loaded in {time.time() - t0:.1f}s on {device}")
    print(f"[Info] LM hidden_size={text_config.hidden_size}, layers={text_config.num_hidden_layers}")

    if lm_head_weight is None:
        print("[Error] lm_head.weight not found in model weights!")
        return

    test_prompts = [
        "What is the capital of France? Answer briefly.",
        "Explain what a neural network is in 2-3 sentences.",
        "Write a short haiku about spring.",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"[Prompt {i+1}] {prompt}")
        print(f"{'='*60}")

        t1 = time.time()
        output = generate_text(
            model, lm_head_weight, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t1
        print(f"[Output] {output}")
        print(f"[Time] {elapsed:.2f}s")

    print("\n[Done] All text generation tests completed.")


if __name__ == "__main__":
    main()
