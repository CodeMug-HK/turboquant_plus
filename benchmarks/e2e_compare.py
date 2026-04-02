"""End-to-end comparison: generate text with full-precision vs TurboQuant-compressed KV cache.

Shows the real-world impact of KV cache compression on actual model output.
After each generation step, we compress and decompress the KV cache,
then feed the lossy version back for the next token prediction.

Usage:
    python3 benchmarks/e2e_compare.py
"""

import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, ".")
from turboquant import TurboQuant, TurboQuantMSE

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS = 30

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function to check if a number is prime.",
]


def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    return model, tokenizer


def compress_decompress_kv(past_kv, k_bits, v_bits, head_dim=128, seed=42):
    """Compress and immediately decompress a HuggingFace DynamicCache.

    k_bits=None means keep K at full precision (simulates q8_0-K).
    v_bits=None means keep V at full precision.
    """
    k_quant = TurboQuant(head_dim, bit_width=k_bits, seed=seed) if k_bits else None
    v_quant = TurboQuantMSE(head_dim, bit_width=v_bits, seed=seed + 500) if v_bits else None

    new_layers = []
    for layer_kv in past_kv:
        layer_tuple = tuple(layer_kv)
        k, v = layer_tuple[0], layer_tuple[1]  # (1, num_heads, seq_len, head_dim)

        if k_quant is not None:
            k_np = k.squeeze(0).numpy()
            k_hat = np.zeros_like(k_np)
            for head in range(k_np.shape[0]):
                compressed_k = k_quant.quantize(k_np[head])
                k_hat[head] = k_quant.dequantize(compressed_k)
            k = torch.from_numpy(k_hat).unsqueeze(0).to(torch.float32)

        if v_quant is not None:
            v_np = v.squeeze(0).numpy()
            v_hat = np.zeros_like(v_np)
            for head in range(v_np.shape[0]):
                v_idx, v_norms = v_quant.quantize(v_np[head])
                v_hat[head] = v_quant.dequantize(v_idx, v_norms)
            v = torch.from_numpy(v_hat).unsqueeze(0).to(torch.float32)

        new_layers.append((k, v))

    cache = DynamicCache()
    for layer_idx, (k_t, v_t) in enumerate(new_layers):
        cache.update(k_t, v_t, layer_idx)
    return cache


def generate_baseline(model, tokenizer, prompt, max_tokens):
    """Generate with full-precision KV cache (normal generation)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def generate_with_turbo(model, tokenizer, prompt, max_tokens, k_bits, v_bits):
    """Generate token-by-token, compressing KV cache after each step."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generated_ids = []
    past_kv = None

    for step in range(max_tokens):
        with torch.no_grad():
            if past_kv is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_ids.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # Compress the KV cache before the next step
        raw_kv = outputs.past_key_values
        head_dim = tuple(raw_kv)[0][0].shape[-1]
        past_kv = compress_decompress_kv(raw_kv, k_bits, v_bits, head_dim=head_dim)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


def token_match_rate(text1, text2, tokenizer):
    """What fraction of tokens match between two outputs."""
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)
    max_len = max(len(tokens1), len(tokens2))
    if max_len == 0:
        return 1.0
    matches = sum(t1 == t2 for t1, t2 in zip(tokens1, tokens2))
    return matches / max_len


def main():
    model, tokenizer = load_model()

    # k_bits=None means keep K at full precision (like q8_0), only compress V
    configs = [
        ("fullprec-K + turbo4-V (safe)",     None, 4),
        ("fullprec-K + turbo3-V (moderate)",  None, 3),
        ("turbo4-K + turbo4-V (symmetric)",   4, 4),
        ("turbo3-K + turbo3-V (aggressive)",  3, 3),
    ]

    print("\n" + "=" * 70)
    print("END-TO-END GENERATION COMPARISON")
    print(f"Model: {MODEL_NAME}  |  Max tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)

    for prompt in PROMPTS:
        print(f"\n{'─' * 70}")
        print(f"PROMPT: {prompt}")
        print(f"{'─' * 70}")

        # Baseline
        t0 = time.perf_counter()
        baseline = generate_baseline(model, tokenizer, prompt, MAX_NEW_TOKENS)
        t_base = time.perf_counter() - t0
        print(f"\n  [BASELINE fp32 KV] ({t_base:.1f}s)")
        print(f"  >>> {baseline[:300]}")

        for name, k_bits, v_bits in configs:
            t0 = time.perf_counter()
            turbo = generate_with_turbo(model, tokenizer, prompt, MAX_NEW_TOKENS, k_bits, v_bits)
            t_turbo = time.perf_counter() - t0
            match = token_match_rate(baseline, turbo, tokenizer)
            print(f"\n  [{name}] ({t_turbo:.1f}s, token match: {match:.0%})")
            print(f"  >>> {turbo[:300]}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
