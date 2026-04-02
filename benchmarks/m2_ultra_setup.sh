#!/bin/bash
# =============================================================================
# Setup script for TurboQuant benchmarks on M2 Ultra 128GB
# =============================================================================
#
# Run this FIRST before m2_ultra_bench.sh.
# Installs all prerequisites, clones the repo, downloads models + test data.
#
# Usage:
#   chmod +x benchmarks/m2_ultra_setup.sh
#   ./benchmarks/m2_ultra_setup.sh
#
# After setup completes, run:
#   ./benchmarks/m2_ultra_bench.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
step() { echo -e "\n${CYAN}── $* ──${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/m2_ultra_workspace"

echo ""
echo "=========================================="
echo "  TurboQuant M2 Ultra Setup"
echo "=========================================="
echo ""

# ─── Check hardware ─────────────────────────────────────────────────────────
step "Checking hardware"

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%.0f", $0/1073741824}')
echo "  Chip: $CHIP"
echo "  RAM:  ${RAM_GB}GB"

if [ "$RAM_GB" -lt 64 ]; then
    warn "Only ${RAM_GB}GB RAM detected. 70B model needs 64GB+. Small model benchmarks will still work."
fi

# ─── Check / install dependencies ───────────────────────────────────────────
step "Checking dependencies"

# cmake
if command -v cmake &>/dev/null; then
    log "cmake $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
else
    err "cmake not found. Installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install cmake
        log "cmake installed"
    else
        err "Homebrew not found. Install cmake manually: https://cmake.org/download/"
        exit 1
    fi
fi

# git
if command -v git &>/dev/null; then
    log "git $(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
else
    err "git not found. Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

# wget or curl
if command -v wget &>/dev/null; then
    log "wget found"
    DOWNLOADER="wget"
elif command -v curl &>/dev/null; then
    log "curl found (will use as fallback)"
    DOWNLOADER="curl"
else
    err "Neither wget nor curl found."
    exit 1
fi

# huggingface-cli (optional, speeds up model downloads)
if command -v huggingface-cli &>/dev/null; then
    log "huggingface-cli found"
    HF_CLI=1
else
    warn "huggingface-cli not found. Will use direct download URLs."
    warn "For faster downloads: pip install huggingface-hub"
    HF_CLI=0
fi

# ─── Raise Metal memory limit ───────────────────────────────────────────────
step "Configuring Metal GPU memory"

if [ "$RAM_GB" -ge 128 ]; then
    WIRED_LIMIT=117964
elif [ "$RAM_GB" -ge 96 ]; then
    WIRED_LIMIT=88474
elif [ "$RAM_GB" -ge 64 ]; then
    WIRED_LIMIT=58982
elif [ "$RAM_GB" -ge 32 ]; then
    WIRED_LIMIT=29491
else
    WIRED_LIMIT=14745
fi

CURRENT=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "0")
if [ "$CURRENT" != "$WIRED_LIMIT" ]; then
    log "Setting Metal wired limit to ${WIRED_LIMIT}MB (90% of ${RAM_GB}GB)"
    sudo sysctl iogpu.wired_limit_mb=$WIRED_LIMIT
else
    log "Metal limit already set to ${WIRED_LIMIT}MB"
fi

# ─── Clone and build llama.cpp ───────────────────────────────────────────────
step "Cloning and building llama.cpp with TurboQuant"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ -d "llama-cpp-turboquant" ]; then
    log "Repo already cloned. Pulling latest..."
    cd llama-cpp-turboquant
    git checkout feature/turboquant-kv-cache
    git pull
else
    log "Cloning from CodeMug-HK fork..."
    git clone https://github.com/CodeMug-HK/llama-cpp-turboquant.git
    cd llama-cpp-turboquant
    git checkout feature/turboquant-kv-cache
fi

if [ -f "build/bin/llama-perplexity" ] && [ -f "build/bin/llama-bench" ] && [ -f "build/bin/llama-cli" ]; then
    log "Build already exists. Skipping build."
else
    log "Building with Metal (using $(sysctl -n hw.ncpu) cores)..."
    cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(sysctl -n hw.ncpu)
    log "Build complete!"
fi

# Verify binaries
for bin in llama-cli llama-server llama-perplexity llama-bench; do
    if [ -f "build/bin/$bin" ]; then
        log "  $bin ✓"
    else
        err "  $bin missing!"
    fi
done

# ─── Download models ─────────────────────────────────────────────────────────
step "Downloading models"

MODELS_DIR="${WORK_DIR}/models"
mkdir -p "$MODELS_DIR"

download_model() {
    local name="$1"
    local dest="$2"
    local hf_repo="$3"
    local hf_file="$4"
    local direct_url="$5"

    if [ -f "$dest" ]; then
        log "$name already downloaded ($(du -h "$dest" | cut -f1))"
        return
    fi

    echo "  Downloading $name..."
    if [ "$HF_CLI" = "1" ]; then
        huggingface-cli download "$hf_repo" "$hf_file" \
            --local-dir "$MODELS_DIR" \
            --local-dir-use-symlinks false 2>/dev/null && {
            log "$name downloaded ($(du -h "$dest" | cut -f1))"
            return
        }
        warn "huggingface-cli failed, trying direct URL..."
    fi

    if [ "$DOWNLOADER" = "wget" ]; then
        wget --progress=bar:force -O "$dest" "$direct_url"
    else
        curl -L --progress-bar -o "$dest" "$direct_url"
    fi
    log "$name downloaded ($(du -h "$dest" | cut -f1))"
}

# Small model: Qwen2.5-1.5B Q8_0 (~1.7GB)
download_model \
    "Qwen2.5-1.5B-Instruct Q8_0 (~1.7GB)" \
    "${MODELS_DIR}/qwen2.5-1.5b-instruct-q8_0.gguf" \
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF" \
    "qwen2.5-1.5b-instruct-q8_0.gguf" \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf"

# Large model: Llama-3.1-70B Q4_K_M (~40GB)
download_model \
    "Llama-3.1-70B-Instruct Q4_K_M (~40GB)" \
    "${MODELS_DIR}/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" \
    "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF" \
    "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" \
    "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"

# ─── Download test data ─────────────────────────────────────────────────────
step "Downloading test data"

WIKITEXT="${WORK_DIR}/wiki.test.raw"
if [ -f "$WIKITEXT" ]; then
    log "wikitext-2 already downloaded"
else
    if [ "$DOWNLOADER" = "wget" ]; then
        wget -q -O "$WIKITEXT" "https://huggingface.co/nisten/llama3-8b-instruct-32k-gguf/raw/main/wiki.test.raw"
    else
        curl -sL -o "$WIKITEXT" "https://huggingface.co/nisten/llama3-8b-instruct-32k-gguf/raw/main/wiki.test.raw"
    fi
    log "wikitext-2 downloaded ($(wc -l < "$WIKITEXT") lines)"
fi

# ─── Quick smoke test ────────────────────────────────────────────────────────
step "Smoke test: quick generation with Qwen2.5-1.5B"

SMALL_MODEL="${MODELS_DIR}/qwen2.5-1.5b-instruct-q8_0.gguf"
if [ -f "$SMALL_MODEL" ]; then
    echo ""
    echo "  Testing baseline (q8_0 KV):"
    cd "${WORK_DIR}/llama-cpp-turboquant"
    ./build/bin/llama-cli \
        -m "$SMALL_MODEL" \
        -ctk q8_0 -ctv q8_0 \
        -fa on -ngl 99 -c 512 \
        -p "The capital of France is" \
        -n 20 --no-display-prompt 2>/dev/null || warn "Smoke test failed (baseline)"

    echo ""
    echo "  Testing turbo3 KV:"
    ./build/bin/llama-cli \
        -m "$SMALL_MODEL" \
        -ctk q8_0 -ctv turbo3 \
        -fa on -ngl 99 -c 512 \
        -p "The capital of France is" \
        -n 20 --no-display-prompt 2>/dev/null || warn "Smoke test failed (turbo3)"
    echo ""
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
step "Setup complete!"

echo ""
echo "  Workspace: $WORK_DIR"
echo "  Models:    $MODELS_DIR"
echo ""
ls -lh "$MODELS_DIR"/ 2>/dev/null
echo ""
echo "  Next steps:"
echo "    1. Run the full benchmark suite:"
echo "       ./benchmarks/m2_ultra_bench.sh"
echo ""
echo "    2. Or try it interactively:"
echo "       cd $WORK_DIR/llama-cpp-turboquant"
echo "       ./build/bin/llama-cli -m $SMALL_MODEL -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 4096"
echo ""
