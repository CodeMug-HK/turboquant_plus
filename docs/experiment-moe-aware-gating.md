# Experiment: MoE-Aware Expert Gating Cache

Branch: `experiment/moe-aware-gating`

## Hypothesis
For MoE models, track which experts fire most frequently. Give frequently-activated experts' KV higher precision. Rarely-used expert KV can tolerate more compression.

## Research Findings

### Expert Routing in Qwen3.5-35B-A3B
- **Architecture:** `LLM_ARCH_QWEN35MOE`, 40 layers
- **Expert selection:** `ggml_argsort_top_k()` on softmax gating probabilities
- **Expert counts:** `n_expert` and `n_expert_used` from GGUF model config
- **Routing tensors available:** `ffn_moe_topk` (selected indices), `ffn_moe_weights` (activation weights), `ffn_moe_probs` (all probabilities)

### Key Insight: KV Cache Is Attention, Not FFN
**Critical problem identified by Codex:** Expert routing is an FFN signal, but the KV cache stores attention projections. The KV vectors are computed BEFORE expert routing — they go through shared attention heads, not per-expert paths.

This means: **KV cache quality is independent of which experts fire.** The expert gating affects FFN outputs, not K/V values. Compressing KV based on expert frequency doesn't make sense because K and V are the same regardless of which experts process the token.

### What WOULD Work
If we wanted MoE-aware compression, the target would be the **FFN intermediate activations** (if cached), not the KV cache. But llama.cpp doesn't cache FFN activations.

Alternatively, if certain tokens consistently route to low-importance experts, those tokens might contribute less to future attention. But measuring "attention importance" requires tracking attention weights, which is separate from expert routing.

### Instrumentation (If We Revisit)
- `cb_eval` callback can capture `ffn_moe_topk` per layer non-intrusively
- Could build expert frequency histogram across a calibration set
- But the fundamental disconnect (FFN signal vs attention cache) remains

## Verdict
**Hypothesis invalid for KV cache.** Expert routing doesn't affect KV values — K and V are computed in the shared attention path before expert routing. This experiment should be **deprioritized** in favor of layer-adaptive and temporal decay.

Codex's assessment was correct: "Expert routing is an FFN signal; KV cache importance is an attention signal. The premise may not map cleanly."

## Status
RESEARCH COMPLETE — hypothesis invalidated. No implementation needed.
