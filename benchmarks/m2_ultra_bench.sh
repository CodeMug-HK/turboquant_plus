#!/bin/bash
# =============================================================================
# TurboQuant Benchmark Suite for M2 Ultra 128GB
# =============================================================================
#
# This script builds llama.cpp with TurboQuant, downloads models,
# and runs perplexity + speed benchmarks at various compression levels.
#
# Usage:
#   chmod +x benchmarks/m2_ultra_bench.sh
#   ./benchmarks/m2_ultra_bench.sh
#
# What it does:
#   1. Clones and builds the TurboQuant llama.cpp fork (Metal)
#   2. Downloads test models (Qwen3-1.7B Q8, Llama-3.1-70B Q4_K_M)
#   3. Downloads wikitext-2 test data
#   4. Runs perplexity baselines and turbo configs
#   5. Runs speed benchmarks (short + long context)
#   6. Outputs all results to benchmarks/results/
#
# Estimated time: ~2-3 hours for all benchmarks
# Estimated disk: ~50GB for models + build
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/m2_ultra_workspace"
RESULTS_DIR="${SCRIPT_DIR}/results"
BUILD_DIR="${WORK_DIR}/llama-cpp-turboquant/build"
BIN="${BUILD_DIR}/bin"

mkdir -p "$RESULTS_DIR"

# ─── Colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
step() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# ─── Step 0: Raise Metal memory limit ───────────────────────────────────────
step "Step 0: Raise Metal GPU memory limit (128GB Mac → 90%)"

WIRED_LIMIT=117964  # 90% of 128GB
CURRENT_LIMIT=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "unknown")
if [ "$CURRENT_LIMIT" != "$WIRED_LIMIT" ]; then
    log "Current limit: ${CURRENT_LIMIT}MB. Setting to ${WIRED_LIMIT}MB..."
    log "This requires sudo. Enter your password if prompted."
    sudo sysctl iogpu.wired_limit_mb=$WIRED_LIMIT
else
    log "Metal memory limit already set to ${WIRED_LIMIT}MB"
fi

# ─── Step 1: Build llama.cpp with TurboQuant ────────────────────────────────
step "Step 1: Build llama.cpp with TurboQuant (Metal)"

if [ -f "${BIN}/llama-perplexity" ] && [ -f "${BIN}/llama-bench" ]; then
    log "Build already exists, skipping. Delete ${WORK_DIR} to rebuild."
else
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"

    if [ ! -d "llama-cpp-turboquant" ]; then
        log "Cloning TurboQuant fork..."
        git clone https://github.com/CodeMug-HK/llama-cpp-turboquant.git
        cd llama-cpp-turboquant
        git checkout feature/turboquant-kv-cache
    else
        cd llama-cpp-turboquant
        git checkout feature/turboquant-kv-cache
        git pull
    fi

    log "Building with Metal..."
    cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(sysctl -n hw.ncpu)
    log "Build complete."
fi

# ─── Step 2: Download models ────────────────────────────────────────────────
step "Step 2: Download models"

MODELS_DIR="${WORK_DIR}/models"
mkdir -p "$MODELS_DIR"

# --- Small model: Qwen2.5-1.5B Q8_0 (~1.7GB) ---
SMALL_MODEL="${MODELS_DIR}/qwen2.5-1.5b-instruct-q8_0.gguf"
if [ ! -f "$SMALL_MODEL" ]; then
    log "Downloading Qwen2.5-1.5B Q8_0..."
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
        qwen2.5-1.5b-instruct-q8_0.gguf \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks false 2>/dev/null || \
    wget -q -O "$SMALL_MODEL" \
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"
    log "Downloaded: $(du -h "$SMALL_MODEL" | cut -f1)"
else
    log "Small model already downloaded: $(du -h "$SMALL_MODEL" | cut -f1)"
fi

# --- Large model: Llama-3.1-70B Q4_K_M (~40GB) ---
LARGE_MODEL="${MODELS_DIR}/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
if [ ! -f "$LARGE_MODEL" ]; then
    log "Downloading Llama-3.1-70B Q4_K_M (~40GB, this will take a while)..."
    huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
        Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
        --local-dir "$MODELS_DIR" \
        --local-dir-use-symlinks false 2>/dev/null || \
    warn "Auto-download failed. Download manually from HuggingFace and place at: $LARGE_MODEL"
else
    log "Large model already downloaded: $(du -h "$LARGE_MODEL" | cut -f1)"
fi

# ─── Step 3: Download test data ─────────────────────────────────────────────
step "Step 3: Download wikitext-2 test data"

WIKITEXT="${WORK_DIR}/wiki.test.raw"
if [ ! -f "$WIKITEXT" ]; then
    log "Downloading wikitext-2..."
    wget -q -O "$WIKITEXT" \
        "https://huggingface.co/nisten/llama3-8b-instruct-32k-gguf/raw/main/wiki.test.raw"
    log "Downloaded: $(wc -l < "$WIKITEXT") lines"
else
    log "Wikitext already downloaded."
fi

# ─── Helper: run perplexity benchmark ────────────────────────────────────────
run_ppl() {
    local label="$1"
    local model="$2"
    local ctk="$3"
    local ctv="$4"
    local ctx="$5"
    local chunks="$6"
    local outfile="${RESULTS_DIR}/${label}.txt"

    log "PPL: ${label} (ctk=${ctk}, ctv=${ctv}, ctx=${ctx}, chunks=${chunks})"
    "${BIN}/llama-perplexity" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -fa on -ngl 99 \
        -f "$WIKITEXT" \
        -c "$ctx" --chunks "$chunks" \
        2>&1 | tee "$outfile"

    # Extract final PPL line
    grep -i "perplexity" "$outfile" | tail -1 >> "${RESULTS_DIR}/summary.txt"
    echo "  ^^^ ${label}" >> "${RESULTS_DIR}/summary.txt"
}

# ─── Helper: run speed benchmark ─────────────────────────────────────────────
run_speed() {
    local label="$1"
    local model="$2"
    local ctk="$3"
    local ctv="$4"
    local prompt_len="$5"
    local outfile="${RESULTS_DIR}/speed_${label}.txt"

    log "Speed: ${label} (ctk=${ctk}, ctv=${ctv}, prompt=${prompt_len})"
    "${BIN}/llama-bench" \
        -m "$model" \
        -ctk "$ctk" -ctv "$ctv" \
        -fa 1 -ngl 99 \
        -p "$prompt_len" -n 128 -r 3 \
        2>&1 | tee "$outfile"

    tail -3 "$outfile" >> "${RESULTS_DIR}/speed_summary.txt"
    echo "  ^^^ ${label}" >> "${RESULTS_DIR}/speed_summary.txt"
}

# ─── Step 4: Small model benchmarks (Qwen2.5-1.5B Q8_0) ────────────────────
step "Step 4: Qwen2.5-1.5B Q8_0 — Perplexity benchmarks"

echo "=== PERPLEXITY RESULTS ===" > "${RESULTS_DIR}/summary.txt"
echo "Date: $(date)" >> "${RESULTS_DIR}/summary.txt"
echo "Machine: M2 Ultra 128GB" >> "${RESULTS_DIR}/summary.txt"
echo "" >> "${RESULTS_DIR}/summary.txt"

if [ -f "$SMALL_MODEL" ]; then
    echo "--- Qwen2.5-1.5B Q8_0 ---" >> "${RESULTS_DIR}/summary.txt"

    # Baseline
    run_ppl "qwen1.5b_q8_baseline"    "$SMALL_MODEL" q8_0   q8_0   512 20

    # Asymmetric (safe — V only compressed)
    run_ppl "qwen1.5b_q8k_turbo4v"    "$SMALL_MODEL" q8_0   turbo4 512 20
    run_ppl "qwen1.5b_q8k_turbo3v"    "$SMALL_MODEL" q8_0   turbo3 512 20
    run_ppl "qwen1.5b_q8k_turbo2v"    "$SMALL_MODEL" q8_0   turbo2 512 20

    # Symmetric
    run_ppl "qwen1.5b_turbo4_turbo4"  "$SMALL_MODEL" turbo4 turbo4 512 20
    run_ppl "qwen1.5b_turbo3_turbo3"  "$SMALL_MODEL" turbo3 turbo3 512 20
fi

# ─── Step 5: Large model benchmarks (Llama-3.1-70B Q4_K_M) ──────────────────
step "Step 5: Llama-3.1-70B Q4_K_M — Perplexity benchmarks"

if [ -f "$LARGE_MODEL" ]; then
    echo "" >> "${RESULTS_DIR}/summary.txt"
    echo "--- Llama-3.1-70B Q4_K_M ---" >> "${RESULTS_DIR}/summary.txt"

    # Baseline
    run_ppl "llama70b_q4_baseline"     "$LARGE_MODEL" q8_0   q8_0   512 20

    # Asymmetric
    run_ppl "llama70b_q8k_turbo4v"     "$LARGE_MODEL" q8_0   turbo4 512 20
    run_ppl "llama70b_q8k_turbo3v"     "$LARGE_MODEL" q8_0   turbo3 512 20
    run_ppl "llama70b_q8k_turbo2v"     "$LARGE_MODEL" q8_0   turbo2 512 20

    # Symmetric (should work on 70B)
    run_ppl "llama70b_turbo4_turbo4"   "$LARGE_MODEL" turbo4 turbo4 512 20
    run_ppl "llama70b_turbo3_turbo3"   "$LARGE_MODEL" turbo3 turbo3 512 20

    # Long context PPL (8K)
    run_ppl "llama70b_q8_baseline_8k"  "$LARGE_MODEL" q8_0   q8_0   8192 10
    run_ppl "llama70b_turbo3_turbo3_8k" "$LARGE_MODEL" turbo3 turbo3 8192 10
fi

# ─── Step 6: Speed benchmarks ───────────────────────────────────────────────
step "Step 6: Speed benchmarks"

echo "=== SPEED RESULTS ===" > "${RESULTS_DIR}/speed_summary.txt"
echo "Date: $(date)" >> "${RESULTS_DIR}/speed_summary.txt"
echo "" >> "${RESULTS_DIR}/speed_summary.txt"

if [ -f "$SMALL_MODEL" ]; then
    echo "--- Qwen2.5-1.5B Q8_0 (speed) ---" >> "${RESULTS_DIR}/speed_summary.txt"
    run_speed "qwen1.5b_q8_short"      "$SMALL_MODEL" q8_0   q8_0   512
    run_speed "qwen1.5b_turbo4_short"  "$SMALL_MODEL" turbo4 turbo4 512
    run_speed "qwen1.5b_turbo3_short"  "$SMALL_MODEL" turbo3 turbo3 512
    run_speed "qwen1.5b_q8_long"       "$SMALL_MODEL" q8_0   q8_0   8192
    run_speed "qwen1.5b_turbo3_long"   "$SMALL_MODEL" turbo3 turbo3 8192
fi

if [ -f "$LARGE_MODEL" ]; then
    echo "" >> "${RESULTS_DIR}/speed_summary.txt"
    echo "--- Llama-3.1-70B Q4_K_M (speed) ---" >> "${RESULTS_DIR}/speed_summary.txt"

    # Short context
    run_speed "llama70b_q8_short"      "$LARGE_MODEL" q8_0   q8_0   512
    run_speed "llama70b_turbo4_short"  "$LARGE_MODEL" turbo4 turbo4 512
    run_speed "llama70b_turbo3_short"  "$LARGE_MODEL" turbo3 turbo3 512

    # Long context — this is where TurboQuant shines
    run_speed "llama70b_q8_8k"         "$LARGE_MODEL" q8_0   q8_0   8192
    run_speed "llama70b_turbo3_8k"     "$LARGE_MODEL" turbo3 turbo3 8192
    run_speed "llama70b_q8_32k"        "$LARGE_MODEL" q8_0   q8_0   32768
    run_speed "llama70b_turbo3_32k"    "$LARGE_MODEL" turbo3 turbo3 32768
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
step "Done! Results saved to ${RESULTS_DIR}/"

echo ""
echo "Files:"
ls -la "${RESULTS_DIR}/"
echo ""
echo "=== PERPLEXITY SUMMARY ==="
cat "${RESULTS_DIR}/summary.txt"
echo ""
echo "=== SPEED SUMMARY ==="
cat "${RESULTS_DIR}/speed_summary.txt"
