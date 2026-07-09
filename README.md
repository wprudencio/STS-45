# Realtime Orb (Kokoro TTS)

Hands-free speech-to-speech voice assistant. Tap the orb, talk, it replies.

```
Browser mic ──16kHz PCM──► ws :7778 ──► VAD → parakeet → llama.cpp → Kokoro-82M
                          (binary)                                 ↓ per clause
Browser speaker ◄──PCM─────────────────────────────────────────────┘
```

Everything runs locally. Use headphones to avoid echo.

## Quick Start

```bash
# 1. Create venv and install deps (Python 3.12 required — torch on Intel Mac)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "numpy<2"

# 2. Download STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-f16.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf

# 3. Place the parakeet-server binary (one-time)
#    Grab the macOS binary from:
#    https://github.com/mudler/parakeet.cpp/releases
#    Extract and place it in: bin/parakeet-server

# 4. Start llama.cpp on the host
llama-server -m your-model.gguf --port 8080

# 5. Start everything else
./run.sh

# 6. Open http://localhost:7777  (or http://0.0.0.0:7777)
```

Or override defaults:

```bash
PORT=9999 LLM_API=http://other-host:8080/v1/chat/completions ./run.sh
```

## Structure

```
├── server.py          Flask: one page + WS realtime startup
├── realtime.py        WebSocket realtime pipeline (STT/LLM/TTS)
├── static/
│   ├── app.js         WS connect, mic capture, audio playback, settings
│   └── styles.css     neon orb UI
├── templates/
│   └── index.html     full-page orb + transcript + settings drawer
├── bin/               parakeet-server binary (gitignored)
├── models/            STT models (gitignored)
├── run.sh             startup script
├── requirements.txt
└── .venv/             Python 3.12 venv (gitignored)
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `:8080` |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) (StyleTTS2, 82M params) | — |
| UI | Flask + vanilla JS | `:7777` (HTTP) · `:7778` (WS) |

## Voices

20+ Kokoro voices across 8 languages: AF Heart, AM Adam, BF Emma, BM George,
EF Dora, FF Siwis, IF Sara, JF Alpha, PF Dora, ZF Xiaobei, and more. Select
from the Settings drawer (gear icon, top-right).

Speed is adjustable (0.5×–2.0×). Language changes also switch the Kokoro model
so voices from the selected language are used.

First launch downloads the Kokoro-82M model (~300MB). Subsequent starts are instant.
