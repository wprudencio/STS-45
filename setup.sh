#!/usr/bin/env bash
# =============================================================================
# STS-45 Local Setup (no Docker)
# =============================================================================
# Installs everything locally: nginx, Python deps, llama.cpp, parakeet.cpp,
# downloads models, and configures the unified reverse proxy on port 7777.
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# After setup, run:  ./start.sh
# Open: http://localhost:7777
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Color output ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Check prerequisites ────────────────────────────────────────────────────
info "Checking prerequisites..."

if [[ $EUID -eq 0 ]]; then
    err "This script should NOT be run as root (sudo will be used when needed)."
    exit 1
fi

# Check OS
if ! grep -qi "ubuntu\|debian" /etc/os-release 2>/dev/null; then
    warn "This script is designed for Ubuntu/Debian. You may need to adapt package names."
fi

# Check Python
PYTHON=""
for cmd in python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    err "Python 3 is required. Install it with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
PYVER=$($PYTHON --version 2>&1)
ok "Found $PYVER"

# Check for sudo
if ! command -v sudo &>/dev/null; then
    err "sudo is required."
    exit 1
fi

# ─── Install system packages ─────────────────────────────────────────────────
info "Installing system packages (nginx, build tools, etc.)..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    nginx \
    curl \
    wget \
    ca-certificates \
    libgomp1 \
    python3-pip \
    python3-venv \
    build-essential \
    git

ok "System packages installed"

# ─── Create Python virtual environment ───────────────────────────────────────
if [[ ! -d .venv ]]; then
    info "Creating Python virtual environment..."
    $PYTHON -m venv .venv
    ok "Virtual environment created"
else
    info "Virtual environment already exists"
fi

source .venv/bin/activate

info "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Python dependencies installed"

# ─── Create directories ──────────────────────────────────────────────────────
mkdir -p bin models models/piper

# ─── Install llama.cpp server ───────────────────────────────────────────────
if ! command -v llama-server &>/dev/null; then
    info "Installing llama.cpp via llama.app installer..."
    curl -LsSf https://llama.app/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    # Add to shell config if not already there
    if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    ok "llama.cpp installed"
else
    info "llama.cpp already installed"
fi

# ─── Download LLM model ──────────────────────────────────────────────────────
MODEL_FILE="LFM2.5-230M-Q4_K_M.gguf"
HF_REPO="LiquidAI/LFM2.5-230M-GGUF"
MODEL_PATH="models/$MODEL_FILE"

if [[ ! -f "$MODEL_PATH" ]]; then
    info "Downloading $MODEL_FILE from $HF_REPO (≈150 MB)..."
    cd models
    wget -q --show-progress "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE" || {
        err "Failed to download model. Check the URL or try manually."
        exit 1
    }
    cd "$SCRIPT_DIR"
    ok "LLM model downloaded"
else
    info "LLM model already present: $MODEL_PATH"
fi

# ─── Download parakeet-server binary ─────────────────────────────────────────
PARAKEET_VERSION="v0.4.0"
PARAKEET_BIN="bin/parakeet-server"

if [[ ! -f "$PARAKEET_BIN" ]]; then
    info "Downloading parakeet-server $PARAKEET_VERSION binary..."
    PARAKEET_URL="https://github.com/mudler/parakeet.cpp/releases/download/${PARAKEET_VERSION}/parakeet-${PARAKEET_VERSION}-bin-linux-cpu-x64.tar.gz"
    cd /tmp
    curl -sL "$PARAKEET_URL" -o parakeet.tar.gz
    # Extract just the parakeet-server binary (it's inside a versioned directory)
    PARAKEET_DIR="parakeet-${PARAKEET_VERSION}-bin-linux-cpu-x64"
    tar xzf parakeet.tar.gz "${PARAKEET_DIR}/parakeet-server"
    cp "${PARAKEET_DIR}/parakeet-server" "$SCRIPT_DIR/bin/"
    chmod +x "$SCRIPT_DIR/bin/parakeet-server"
    rm -rf parakeet.tar.gz "$PARAKEET_DIR"
    cd "$SCRIPT_DIR"
    ok "parakeet-server binary installed"
else
    info "parakeet-server binary already present"
fi

# ─── Download STT model ──────────────────────────────────────────────────────
STT_MODEL_FILE="tdt_ctc-110m-f16.gguf"
STT_MODEL_PATH="models/$STT_MODEL_FILE"

if [[ ! -f "$STT_MODEL_PATH" ]]; then
    info "Downloading STT model $STT_MODEL_FILE..."
    cd models
    wget -q --show-progress "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/$STT_MODEL_FILE" || {
        warn "Primary STT model not found, trying alternative..."
        wget -q --show-progress "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-1.1b-q5_k.gguf" || {
            err "Failed to download STT model."
            exit 1
        }
    }
    cd "$SCRIPT_DIR"
    ok "STT model downloaded"
else
    info "STT model already present: $STT_MODEL_PATH"
fi

# ─── Configure nginx ─────────────────────────────────────────────────────────
NGINX_CONF_SRC="docker/nginx/default.conf"
NGINX_CONF_DST="/etc/nginx/sites-available/sts45"
NGINX_ENABLED="/etc/nginx/sites-enabled/sts45"

info "Configuring nginx reverse proxy..."
sudo cp "$NGINX_CONF_SRC" "$NGINX_CONF_DST"

# Remove default site if it exists and conflicts
if [[ -f /etc/nginx/sites-enabled/default ]]; then
    sudo rm -f /etc/nginx/sites-enabled/default
fi

# Enable sts45 site
if [[ ! -f "$NGINX_ENABLED" ]]; then
    sudo ln -sf "$NGINX_CONF_DST" "$NGINX_ENABLED"
fi

# Test nginx config
sudo nginx -t || {
    err "Nginx configuration test failed."
    exit 1
}

# Restart nginx to pick up changes
sudo systemctl restart nginx || sudo nginx -s reload 2>/dev/null || true
ok "Nginx configured and restarted on port 7777"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ✅ STS-45 setup complete!                                   ║"
echo "║                                                               ║"
echo "║   IMPORTANT: Edit the start.sh if needed, then run:            ║"
echo "║                                                               ║"
echo "║       ./start.sh                                             ║"
echo "║                                                               ║"
echo "║   Then open:  http://localhost:7777                           ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Services:"
echo "  Nginx reverse proxy  →  http://localhost:7777  (single URL)"
echo "  Flask app (internal) →  http://localhost:7778"
echo "  WS server (internal) →  ws://localhost:7779"
echo "  llama-server (LLM)   →  http://localhost:8080"
echo "  parakeet-server (STT)→  http://localhost:8081"
echo ""
echo "Models:"
echo "  LLM:  models/LFM2.5-230M-Q4_K_M.gguf"
echo "  STT:  models/tdt_ctc-110m-f16.gguf"
echo "  TTS:  models/piper/ (auto-downloaded on first use)"
echo ""
echo "Before starting, make sure llama-server is running:"
echo "  llama-server serve -m models/LFM2.5-230M-Q4_K_M.gguf --host 127.0.0.1 --port 8080"
echo "  (or: ~/.llama-app/llama serve -m models/LFM2.5-230M-Q4_K_M.gguf --host 127.0.0.1 --port 8080)"
echo ""
echo "Then run:  ./start.sh"
echo ""
echo "Open: http://localhost:7777"
echo ""
