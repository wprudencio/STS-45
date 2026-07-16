#!/usr/bin/env bash
# =============================================================================
# STS-45 Start (local, no Docker)
# =============================================================================
# Starts: parakeet STT → Flask + WS server → nginx
# Open: http://localhost:7777
#
# Optional Cloudflare Tunnel (public URL):
#   ./start.sh --cf           # random *.trycloudflare.com URL
#   ./start.sh --cf my.domain # your own domain (requires configured tunnel)
#   CLOUDFLARE=1 ./start.sh   # same as --cf
#
# First run: ./setup.sh  (one-time)
# Then run:  ./start.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ──────────────────────────────────────────────────────────────────
STT_PORT=8081
STT_MODEL="models/tdt_ctc-110m-f16.gguf"
# Flask HTTP on :7778, WS on :7779, nginx on :7777 proxies both
CHAT_HOST="127.0.0.1"
CHAT_PORT=7778
WS_PORT=7779
LLM_API="${LLM_API:-http://127.0.0.1:8080/v1/chat/completions}"
LLM_MODEL="${LLM_MODEL:-default}"

# Cloudflare Tunnel mode
#   --cf              quick tunnel (random *.trycloudflare.com)
#   --cf my.domain    named tunnel (requires ~/.cloudflared/sts45.yml)
#   CLOUDFLARE=1      same as --cf
CLOUDFLARE="${CLOUDFLARE:-}"
CLOUDFLARE_DOMAIN=""
for arg in "$@"; do
    case "$arg" in
        --cf|--cloudflare)
            CLOUDFLARE=1
            ;;
        --cf=*|--cloudflare=*)
            CLOUDFLARE=1
            CLOUDFLARE_DOMAIN="${arg#*=}"
            ;;
        *)
            if [[ -n "$CLOUDFLARE" && -z "$CLOUDFLARE_DOMAIN" ]]; then
                CLOUDFLARE_DOMAIN="$arg"
            fi
            ;;
    esac
done

# Tell the template/JS that the WS is reachable on the same port as nginx (:7777)
export WS_CLIENT_PORT=7777

cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    if [[ -n "${CF_PID:-}" ]]; then
        kill $CF_PID 2>/dev/null || true
        wait $CF_PID 2>/dev/null || true
    fi
    kill $CHAT_PID 2>/dev/null || true
    kill $STT_PID 2>/dev/null || true
    kill $LLAMA_PID 2>/dev/null || true
    wait $CHAT_PID 2>/dev/null || true
    wait $STT_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
    echo "✅ All stopped."
}
trap cleanup EXIT INT TERM

# Kill any stale instances
pkill -f "python3.*server.py" 2>/dev/null || true
pkill -f "parakeet-server"    2>/dev/null || true
LLAMA_PID=""

# ─── Activate venv ───────────────────────────────────────────────────────────
PYTHON="python3"
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
fi

# ─── STT (parakeet) ──────────────────────────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/bin/parakeet-server" ]]; then
    echo "❌ parakeet-server binary not found in bin/."
    echo "   Run: ./setup.sh"
    exit 1
fi

if [[ ! -f "$STT_MODEL" ]]; then
    echo "❌ STT model not found: $STT_MODEL"
    echo "   Run: ./setup.sh"
    exit 1
fi

echo "🎙️  Starting parakeet STT on :$STT_PORT..."
"$SCRIPT_DIR/bin/parakeet-server" --model "$STT_MODEL" --port "$STT_PORT" &
STT_PID=$!
sleep 2

if ! kill -0 $STT_PID 2>/dev/null; then
    echo "❌ STT failed to start."
    echo "   Check that libgomp1 is installed: sudo apt install libgomp1"
    exit 1
fi

# ─── LLM (llama.cpp) ────────────────────────────────────────────────────────
# Check if llama-server is already running on port 8080
if ! curl -s --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "🔍 Looking for llama-server..."
    LLAMA_BIN=""
    for p in "$HOME/.local/bin/llama-server" "/usr/local/bin/llama-server" \
             "/root/.local/bin/llama-server" "/home/musicweslei/.llama-app/llama" \
             "$(which llama-server 2>/dev/null)" "$(which llama 2>/dev/null)"; do
        if [[ -x "$p" ]]; then
            LLAMA_BIN="$p"
            break
        fi
    done

    if [[ -n "$LLAMA_BIN" ]]; then
        LLM_MODEL_PATH="$SCRIPT_DIR/models/gemma-4-E2B-UD-Q2.gguf"
        if [[ -f "$LLM_MODEL_PATH" ]]; then
            echo "🚀 Starting llama-server from $LLAMA_BIN on :8080..."
            # New llama.app uses subcommands: 'llama-server serve' or 'llama serve'
            LLAMA_ARGS="-m \"$LLM_MODEL_PATH\" --host 127.0.0.1 --port 8080 --no-kv-offload -c 2048"
            if echo "$LLAMA_BIN" | grep -q "llama$" || [[ "$(basename "$LLAMA_BIN")" == "llama" ]]; then
                eval "\"$LLAMA_BIN\" serve $LLAMA_ARGS &"
            else
                eval "\"$LLAMA_BIN\" $LLAMA_ARGS &"
            fi
            LLAMA_PID=$!
            echo "   PID: $LLAMA_PID"
            # Wait for it to be ready (can take a while on first load)
            echo "   Waiting for llama-server to load the model..."
            for i in $(seq 1 60); do
                if curl -s --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
                    echo "✅ llama-server ready (after ${i}s)"
                    break
                fi
                sleep 1
            done
        else
            echo "❌ LLM model not found at $LLM_MODEL_PATH"
            echo "   Run: ./setup.sh"
        fi
    else
        echo "⚠️  llama-server not found in PATH."
        echo "   Install: curl -LsSf https://llama.app/install.sh | sh"
        echo "   Or start manually: llama-server -m models/gemma-4-E2B-UD-Q2.gguf --host 127.0.0.1 --port 8080"
    fi
else
    echo "✅ llama-server already running on :8080"
fi

# ─── Start nginx if not running ─────────────────────────────────────────────
if ! systemctl is-active --quiet nginx 2>/dev/null; then
    echo "🔄 Starting nginx..."
    sudo systemctl start nginx 2>/dev/null || sudo nginx 2>/dev/null || true
fi

# ─── Realtime server (Flask + WS) ────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║                                                       ║"
echo "║   🚀 Starting STS-45 (Piper TTS)...                   ║"
echo "║                                                       ║"
echo "║   Open:  http://localhost:7777                        ║"
echo "║                                                       ║"
echo "║   Nginx :7777 → Flask :7778 + WS :7779               ║"
echo "║   STT:   http://localhost:$STT_PORT                     ║"
echo "║   LLM:   $LLM_API                                      ║"
echo "║                                                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Use the Python from the venv (if available) or system Python
if [[ -f "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
    # Activate venv for library path
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

$PYTHON server.py \
    --host "$CHAT_HOST" \
    --port "$CHAT_PORT" \
    --ws-port "$WS_PORT" \
    --stt-api "http://localhost:$STT_PORT" \
    --api "$LLM_API" \
    --model "$LLM_MODEL" &
CHAT_PID=$!

echo "   Flask PID: $CHAT_PID"
echo ""

# ─── Cloudflare Tunnel ───────────────────────────────────────────────────────
if [[ -n "$CLOUDFLARE" ]]; then
    if command -v cloudflared &>/dev/null; then
        # Wait a moment for the Flask server to be fully ready
        sleep 2
        echo "☁️  Starting Cloudflare Tunnel..."
        if [[ -n "$CLOUDFLARE_DOMAIN" ]]; then
            # Named tunnel mode — expects config at ~/.cloudflared/sts45.yml
            echo "   Using tunnel config for domain: $CLOUDFLARE_DOMAIN"
            echo "   (Make sure ~/.cloudflared/sts45.yml exists)"
            nohup cloudflared tunnel --config ~/.cloudflared/sts45.yml run > /tmp/cloudflared.log 2>&1 &
            CF_PID=$!
            echo "   ☁️  Tunnel PID: $CF_PID"
            echo "   📡 https://$CLOUDFLARE_DOMAIN"
        else
            # Quick tunnel — random *.trycloudflare.com URL
            echo "   Starting quick tunnel (random URL)..."
            nohup cloudflared tunnel --url http://localhost:7777 > /tmp/cloudflared.log 2>&1 &
            CF_PID=$!
            echo "   ☁️  Tunnel PID: $CF_PID"
            # Wait for the tunnel URL to appear in the log
            for i in $(seq 1 15); do
                CF_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -1)
                if [[ -n "$CF_URL" ]]; then
                    echo "   📡 Public URL: $CF_URL"
                    break
                fi
                sleep 1
            done
            if [[ -z "${CF_URL:-}" ]]; then
                echo "   ⏳ Tunnel starting... check /tmp/cloudflared.log"
            fi
        fi
        echo ""
    else
        echo "⚠️  cloudflared not installed. Run: ./setup.sh"
    fi
fi

wait $CHAT_PID
