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
python3.12 -m venv lite/.venv
source lite/.venv/bin/activate
pip install -r lite/requirements.txt
pip install "numpy<2"

# 2. link STT binary and models (already done if you set up the full version)
# lite/bin -> ../bin   lite/models -> ../models

# 3. Start llama.cpp on the host
llama-server -m your-model.gguf --port 8080

# 4. Start everything else
./lite/run.sh

# 5. Open http://localhost:7777  (or http://0.0.0.0:7777)
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
