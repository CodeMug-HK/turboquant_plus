# TurboQuant Evaluation Report: Google's Claims vs Our Test Results
# TurboQuant 評估報告：Google 的聲稱 vs 我們的實測結果

**Date / 日期:** 2026-04-02
**Test Hardware / 測試硬體:** Apple M4 Pro, 24GB RAM
**Test Model / 測試模型:** Qwen2.5-1.5B-Instruct Q8_0 (1.8GB)
**Source Repo / 原始 Repo:** [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
**Our Fork / 我們的 Fork:** [CodeMug-HK/turboquant_plus](https://github.com/CodeMug-HK/turboquant_plus)
**Google Research Blog / Google 研究博客:** [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
**Paper / 論文:** [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)

---

## Executive Summary / 摘要

TurboQuant is a real, working algorithm from Google Research (ICLR 2026). Our hands-on testing confirms its core scientific claims are **partially true** — but the marketing language significantly overstates what typical users will experience.

TurboQuant 是 Google Research 的真實算法（ICLR 2026）。我們的實測確認其核心科學聲稱是**部分正確**的——但其宣傳語言明顯誇大了一般用戶的實際體驗。

**Verdict / 結論: 7/10 — The science is solid, but "zero accuracy loss" is misleading.**
**評分: 7/10 — 科學是扎實的，但「零精度損失」的說法有誤導性。**

---

## Claim-by-Claim Analysis / 逐項聲稱分析

### Claim 1: "Zero accuracy loss" / 「零精度損失」

**Google says:** "a compression method that achieves a high reduction in model size with zero accuracy loss"

**Google 說：**「一種壓縮方法，在零精度損失的情況下實現模型大小的高度壓縮」

#### Our test results / 我們的測試結果

| Config 配置 | KV Buffer | PPL (Perplexity) | vs Baseline | Accuracy Loss 精度損失 |
|-------------|-----------|-------------------|-------------|------------------------|
| q8_0/q8_0 (baseline) | 29.75 MiB | 11.11 | — | — |
| q8_0-K + turbo4-V | 22.31 MiB | 11.24 | +1.1% | Small 微小 |
| q8_0-K + turbo3-V | 20.34 MiB | 11.37 | +2.4% | Small 微小 |
| q8_0-K + turbo2-V | 20.19 MiB | 11.67 | +5.0% | Moderate 中等 |
| turbo4/turbo4 (symmetric) | 15.00 MiB | 6,746 | +60,614% | Catastrophic 災難性 |
| turbo3/turbo3 (symmetric) | 11.06 MiB | 8,274 | +74,371% | Catastrophic 災難性 |

**Verdict: MISLEADING / 判定：有誤導性**

"Zero accuracy loss" is only true under very specific conditions:
- V-only compression with full-precision K (asymmetric config)
- Large models (27B+) with Q8_0 weights
- turbo4 bit-width

On smaller models or with symmetric compression (which the paper primarily tests), accuracy degradation is measurable (+1-5%) or catastrophic (PPL 6,000+). The blog post does not mention these failure modes at all.

「零精度損失」只在非常特定的條件下成立：
- 僅壓縮 V cache，K 保持全精度（非對稱配置）
- 大模型（27B+）搭配 Q8_0 權重
- turbo4 位寬

在較小的模型或對稱壓縮（論文主要測試的方式）下，精度下降是可測量的（+1-5%）或災難性的（PPL 6,000+）。博客文章完全沒有提到這些失敗模式。

---

### Claim 2: "6x memory reduction with perfect downstream results" / 「6 倍記憶體壓縮，下游結果完美」

**Google says:** "TurboQuant achieves perfect downstream results across all benchmarks while reducing the key value memory size by a factor of at least 6x"

#### Our test results / 我們的測試結果

| Config | KV Compression | Quality |
|--------|---------------|---------|
| q8_0-K + turbo4-V | 1.33x KV saving | Good quality 品質良好 |
| q8_0-K + turbo3-V | 1.46x KV saving | Good quality 品質良好 |
| turbo3/turbo3 | 2.69x KV saving | Depends on model 取決於模型 |
| turbo2/turbo2 (theoretical 6x) | — | Severe degradation 嚴重退化 |

**Repo's documented results from larger models / 大模型的文件記錄結果：**

- Command-R+ 104B turbo3/turbo3 at 128K context: **5.12x compression**, PPL +3.6% — works well
- Llama-70B turbo3/turbo3: **5.12x compression**, PPL +11.4% — usable but noticeable
- Qwen2.5-7B turbo3/turbo3: PPL **3,556** — completely broken

**Verdict: PARTIALLY TRUE / 判定：部分正確**

6x compression is achievable but only on specific large models. The "perfect downstream results" claim is overstated — even the best configs show measurable PPL increases. The 6x number comes from turbo2 (2-bit), which shows significant quality degradation in most practical scenarios. 5.12x via turbo3 is more realistic, and it works well on 70B+ models.

6 倍壓縮是可以實現的，但僅限於特定的大模型。「完美的下游結果」的說法被誇大了——即使最好的配置也會顯示可測量的 PPL 增加。6 倍的數字來自 turbo2（2-bit），在大多數實際場景中會顯示顯著的品質下降。turbo3 的 5.12 倍更為現實，在 70B+ 模型上效果良好。

---

### Claim 3: "4-bit TurboQuant achieves up to 8x performance increase" / 「4-bit TurboQuant 性能提升達 8 倍」

**Google says:** "4-bit TurboQuant achieves up to 8x performance increase over 32-bit unquantized keys on H100 GPU accelerators"

#### Our test results (M4 Pro, Metal) / 我們的測試結果

| Config | Prompt (t/s) | Decode (t/s) | vs Baseline |
|--------|-------------|-------------|-------------|
| q8_0/q8_0 | 2,279 | 111.2 | — |
| q8_0-K + turbo4-V | 2,253 | 89.2 | Decode **-20%** |
| q8_0-K + turbo3-V | 2,242 | 86.1 | Decode **-23%** |

**Repo's documented results from M5 Max / M5 Max 的文件記錄結果：**

- turbo3 prefill is **+7.4% faster** than q8_0 at 32K context on 70B model
- turbo3 decode is **slower** than q8_0 at short context
- Speed advantage only appears at **long context (32K+)** where memory bandwidth is the bottleneck

**Verdict: TECHNICALLY TRUE BUT MISLEADING / 判定：技術上正確但有誤導性**

The 8x claim is specifically for "computing attention logits" on H100 vs **32-bit** (fp32) baseline. This is comparing against an unrealistic baseline — nobody runs fp32 KV cache in production. Compared to the practical baseline (q8_0), TurboQuant is actually **slower** at short context due to dequantization overhead. The speed advantage only materializes at long context (32K+) on large models, where the smaller KV cache means less memory bandwidth pressure.

8 倍的聲稱特指在 H100 上「計算注意力 logits」，且是與 **32-bit**（fp32）基線比較。這是在與一個不現實的基線比較——沒有人在生產環境中使用 fp32 KV cache。與實際基線（q8_0）相比，TurboQuant 在短 context 下實際上**更慢**，因為需要解量化的額外開銷。速度優勢只在大模型的長 context（32K+）下才會顯現，此時較小的 KV cache 意味著更少的記憶體頻寬壓力。

---

### Claim 4: "3-bit quantization without training or fine-tuning" / 「3-bit 量化無需訓練或微調」

**Google says:** "TurboQuant proved it can quantize the key-value cache to just 3 bits without requiring training or fine-tuning"

#### Our test results / 我們的測試結果

This is confirmed. TurboQuant is entirely post-training. You just add `-ctk turbo3 -ctv turbo3` flags and it works. No calibration data, no retraining, no model modification.

這一點確認屬實。TurboQuant 完全是後訓練的。你只需添加 `-ctk turbo3 -ctv turbo3` 標誌就能使用。不需要校準數據、不需要重新訓練、不需要修改模型。

**Verdict: TRUE / 判定：正確**

---

### Claim 5: "Optimal distortion rates in a data-oblivious manner" / 「以數據無關的方式實現最優失真率」

**Google says:** TurboQuant is "data-oblivious" — it doesn't need dataset-specific tuning.

#### Our test results / 我們的測試結果

The algorithm uses random rotations (Walsh-Hadamard Transform) and fixed codebooks derived from Beta distributions. No model-specific or data-specific calibration is needed. This is confirmed.

However, the actual quality is very much **model-dependent**. The same turbo3/turbo3 config gives +1.1% PPL on Qwen3.5-35B but PPL 3,556 on Qwen2.5-7B. "Data-oblivious" algorithm, yes — but definitely not "model-oblivious" results.

算法使用隨機旋轉（Walsh-Hadamard 變換）和基於 Beta 分佈的固定碼本。不需要模型特定或數據特定的校準。這一點確認屬實。

然而，實際品質非常**依賴模型**。相同的 turbo3/turbo3 配置在 Qwen3.5-35B 上 PPL 僅 +1.1%，但在 Qwen2.5-7B 上 PPL 達到 3,556。算法是「數據無關」的，沒錯——但結果絕對不是「模型無關」的。

**Verdict: TRUE (algorithm), INCOMPLETE (practical impact) / 判定：正確（算法層面），不完整（實際影響）**

---

## What Actually Works / 實際有效的部分

Based on our testing + repo's documented community validation across 30+ testers:

基於我們的測試 + repo 中記錄的 30+ 測試者的社區驗證：

### The winning formula / 致勝公式

```
Keep K at full precision + compress V aggressively = free compression
保持 K 全精度 + 激進壓縮 V = 免費壓縮
```

| Scenario 場景 | Config 配置 | Compression 壓縮 | Quality 品質 | Works? |
|---------------|-------------|------------------|-------------|--------|
| Any model, safe default | q8_0-K + turbo4-V | ~1.3x KV | +0.3-1.1% PPL | Always |
| Any model, more compression | q8_0-K + turbo3-V | ~1.5x KV | +1.1-2.4% PPL | Always |
| Large model (27B+) Q8_0 | turbo4/turbo4 | 3.76x KV | +0.2-1.7% PPL | Yes |
| Large model (70B+) Q4_K_M | turbo3/turbo3 | 5.12x KV | +3.6-11.4% PPL | Model-dependent |
| Small model (<7B) symmetric | turbo3/turbo3 | 5.12x KV | PPL explodes | Never |

### The real value / 真正的價值

TurboQuant's killer use case is NOT small models or short context. It's:

TurboQuant 的殺手級用例不是小模型或短 context。而是：

**Running 70B-104B models at 32K-128K context on consumer hardware**
**在消費級硬體上運行 70B-104B 模型，context 長度達 32K-128K**

- Command-R+ 104B at 128K context on M5 Max 128GB — only possible with turbo3 compression
- 104B 模型 + 128K context 在 M5 Max 128GB 上——只有透過 turbo3 壓縮才可能

---

## Text Generation Quality / 文字生成品質

We ran real text generation on our M4 Pro with all three configs. All asymmetric configs (K at full precision) produced fluent, coherent, and factually correct output. No human-detectable quality difference.

我們在 M4 Pro 上進行了真實文字生成。所有非對稱配置（K 保持全精度）都產生了流暢、連貫、事實正確的輸出。沒有人類可察覺的品質差異。

| Config | Output (excerpt) |
|--------|-----------------|
| Baseline q8_0 | "...compressing the data in a way that minimizes storage space without significantly impacting performance..." |
| q8_0-K + turbo4-V | "...encoding data in a more compact form. By compressing the cache, the system can store more data..." |
| q8_0-K + turbo3-V | "...compressing data before storing it in the cache and later decompressing it when retrieval is needed..." |

All three are coherent and correct. The exact wording differs (expected with lossy compression) but the semantic content is equivalent.

三個輸出都是連貫且正確的。確切措辭不同（有損壓縮的預期行為），但語義內容是等價的。

---

## Summary Scorecard / 總結評分卡

| Claim 聲稱 | Verdict 判定 | Score 評分 |
|------------|-------------|-----------|
| Zero accuracy loss 零精度損失 | Misleading — only true for V-only, large models, turbo4 | 4/10 |
| 6x memory reduction 6 倍記憶體壓縮 | Partially true — 5x realistic on large models, 6x degrades quality | 6/10 |
| 8x speed increase 8 倍性能提升 | Misleading — vs unrealistic fp32 baseline, slower at short context | 3/10 |
| No training needed 無需訓練 | True — confirmed, just add CLI flags | 10/10 |
| Data-oblivious 數據無關 | True algorithmically, but results are very model-dependent | 7/10 |
| V compression is free V壓縮是免費的 | True — confirmed in our testing, the most important finding | 9/10 |

**Overall / 總體: 7/10**

The core science (PolarQuant + QJL for near-optimal vector quantization) is real and well-founded. The practical implementation works. But Google's blog post marketing language — "zero accuracy loss", "8x speedup", "perfect results" — overpromises relative to what real users experience. The community implementation (this repo) is more honest about the tradeoffs.

核心科學（PolarQuant + QJL 實現近最優向量量化）是真實且有根據的。實際實現是可用的。但 Google 博客的營銷語言——「零精度損失」、「8 倍加速」、「完美結果」——相對於真實用戶的體驗過度承諾。社區實現（本 repo）對權衡取捨更為誠實。

---

## Recommendation / 建議

**For our M2 Ultra 128GB test:**
- Test with Llama-3.1-70B Q4_K_M at 32K context — this is where TurboQuant should shine
- Use asymmetric config (`q8_0-K + turbo3-V`) as the safe default
- Also test symmetric (`turbo3/turbo3`) — Llama 70B should tolerate it based on repo data
- Compare KV buffer size, PPL, and decode speed at 8K/32K context lengths

**對於我們的 M2 Ultra 128GB 測試：**
- 用 Llama-3.1-70B Q4_K_M 在 32K context 下測試——這是 TurboQuant 應該大放異彩的場景
- 使用非對稱配置（`q8_0-K + turbo3-V`）作為安全預設
- 也測試對稱配置（`turbo3/turbo3`）——根據 repo 數據，Llama 70B 應該能承受
- 比較 8K/32K context 長度下的 KV buffer 大小、PPL 和 decode 速度
