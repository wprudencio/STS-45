# Supertonic Realtime Lite

A lightweight, hands-free **speech-to-speech** voice assistant — just the realtime
conversation orb from the full Supertonic Voice Chat, nothing else. No chat
history, no sidebar, no push-to-talk, no markdown. Open the page, tap the orb,
and talk.

```
Browser mic ──16kHz PCM──► ws :7778 ──► VAD → parakeet → llama.cpp → Supertonic
                          (binary)                                 ↓ TTS per clause
Browser speaker ◄──PCM─────────────────────────────────────────────┘
```

Everything runs locally. Use headphones to avoid speaker→mic echo.

## Quick Start

```bash
# 1. Install Python deps
pip install -r lite/requirements.txt

# 2. Download STT model (one-time)
mkdir -p lite/models
curl -L -o lite/models/tdt_ctc-110m-f16.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf

# 3. Place the parakeet-server binary (one-time)
#    Grab the macOS binary from:
#    https://github.com/mudler/parakeet.cpp/releases
#    Extract and place it in: lite/bin/parakeet-server

# 4. Start llama.cpp on the host
llama-server -m your-model.gguf --port 8080

# 5. Start everything else
./lite/run.sh

# 6. Open http://localhost:7777  (or http://0.0.0.0:7777)
```

Or override defaults:

```bash
PORT=9999 LLM_API=http://other-host:8080/v1/chat/completions ./lite/run.sh
```

## Structure

```
lite/
├── server.py          # Flask: one page + /api/settings + WS realtime startup
├── realtime.py        # WebSocket realtime pipeline (STT/LLM/TTS)
├── static/
│   ├── app.js         # WS connect, mic capture, audio playback, settings
│   └── styles.css     # neon orb UI
├── templates/
│   └── index.html     # full-page orb + transcript + settings drawer
├── bin/               # parakeet-server binary (gitignored)
├── models/            # STT models (gitignored)
├── run.sh             # startup script
└── requirements.txt
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `:8080` |
| TTS | [Supertonic](https://github.com/supertone-inc/supertonic) (ONNX Runtime) | — |
| UI | Flask + vanilla JS | `:7777` (HTTP) · `:7778` (WS) |

## How it works

Tap the orb to start. The browser streams 16 kHz PCM16 to the WebSocket; the
server runs an energy VAD to find utterance boundaries, transcribes each
utterance with parakeet, streams the llama.cpp reply, and speaks it back
sentence-by-sentence as PCM16 frames so audio starts before the full reply is
generated. Interrupt the assistant any time by speaking — barge-in stops
playback.

Voice, language, diffusion steps, speed, max tokens, system prompt, and the
STT/LLM API URLs are all in the **Settings** drawer (gear icon, top-right) and
are shared with the browser via the `/api/settings` endpoint. Settings persist
in `localStorage`.