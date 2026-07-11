#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- config ----------
STT_PORT=8081
STT_MODEL="models/tdt_ctc-110m-f16.gguf"
CHAT_HOST="${HOST:-0.0.0.0}"
CHAT_PORT="${PORT:-7777}"
LLM_API="${LLM_API:-http://127.0.0.1:8080/v1/chat/completions}"
LLM_MODEL="${LLM_MODEL:-default}"

cleanup() {
  echo ""
  echo "🛑 Shutting down..."
  kill $CHAT_PID 2>/dev/null
  kill $STT_PID 2>/dev/null
  wait $CHAT_PID 2>/dev/null
  wait $STT_PID 2>/dev/null
  echo "✅ All stopped."
}
trap cleanup EXIT INT TERM

# ---------- kill any stale instances ----------
pkill -f "python3.*server.py" 2>/dev/null || true
pkill -f "parakeet-server"    2>/dev/null || true
sleep 1

# ---------- STT (parakeet) ----------
if [ ! -f "./bin/parakeet-server" ]; then
  echo "❌ parakeet-server binary not found in bin/."
  echo "   Download from: https://github.com/mudler/parakeet.cpp/releases"
  echo "   Extract and place the binary in: ./bin/parakeet-server"
  exit 1
fi

if [ ! -f "$STT_MODEL" ]; then
  echo "❌ STT model not found: $STT_MODEL"
  echo "   Run: mkdir -p models && curl -L -o $STT_MODEL https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-1.1b-q5_k.gguf"
  exit 1
fi

echo "🎙️  Starting parakeet STT on :$STT_PORT..."
./bin/parakeet-server --model "$STT_MODEL" --port "$STT_PORT" &
STT_PID=$!
sleep 2

if ! kill -0 $STT_PID 2>/dev/null; then
  echo "❌ STT failed to start."
  exit 1
fi

# ---------- Realtime server (Flask + WS) ----------
echo "🚀 Starting STS-45 (Piper TTS)..."
echo "   Open: http://$CHAT_HOST:$CHAT_PORT  (also: http://localhost:$CHAT_PORT)"
echo "   WS:   ws://$CHAT_HOST:$((CHAT_PORT + 1))/ws"
echo "   STT:  http://localhost:$STT_PORT"
echo "   LLM:  $LLM_API"
echo ""

if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
  PYTHON=python3
fi

$PYTHON server.py \
  --host "$CHAT_HOST" \
  --port "$CHAT_PORT" \
  --stt-api "http://localhost:$STT_PORT" \
  --api "$LLM_API" \
  --model "$LLM_MODEL" &
CHAT_PID=$!

wait $CHAT_PID