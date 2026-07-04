# Supertonic — Voice Chat

Local voice assistant: mic → STT → LLM → TTS → speaker. Everything on-device.

```
Browser mic ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
               (STT)            (LLM)         (ONNX Runtime)
```

## Quick Start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Download STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q5_k.gguf

# 3. Download parakeet-server binary (one-time)
#    Grab the macOS binary from:
#    https://github.com/mudler/parakeet.cpp/releases
#    Extract and place it in: bin/parakeet-server

# 4. Start llama.cpp on host
llama-server -m your-model.gguf --port 8080

# 5. Start everything else
./run.sh

# 6. Open http://localhost:7777
```

Or override defaults:

```bash
PORT=9999 LLM_API=http://other-host:8080/v1/chat/completions ./run.sh
```

## Project structure

```
supertonic/
├── app/
│   ├── chat_ui.py          # Flask API + main entrypoint
│   ├── realtime.py         # WebSocket realtime server
│   ├── static/             # CSS, JS
│   └── templates/          # HTML
├── bin/                    # parakeet-server binary (gitignored)
├── models/                 # STT models (gitignored)
├── run.sh                  # startup script
└── requirements.txt
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (NVIDIA NeMo → ggml) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `:8080` |
| TTS | [Supertonic 3](https://github.com/supertone-inc/supertonic) (ONNX Runtime) | — |
| UI | Flask + vanilla JS | `:7777` |

## Usage

Click the mic button (or double-tap `Shift`) to talk; `Shift` once stops the
mic, `Enter` sends. Edit voice, language, and system prompt from the **Settings**
dialog and the per-conversation system-prompt strip.

Extra features: stop/regenerate a response, edit & resend a previous user
message, branch a conversation from any message, pin messages, export/import
(Markdown or JSON), prompt templates (type `/`), live token & latency stats,
and a keyboard-shortcut overlay (`?` or `⌘⇧O`). All conversation history is
stored locally in your browser (IndexedDB).

## Realtime conversation mode

Tap the **mic icon in the top-right** to start a hands-free, speech-to-speech
conversation — like the [HF realtime voice](https://huggingface.co/spaces/smolagents/hf-realtime-voice)
demo, but fully local. Just talk; the assistant listens (VAD), transcribes
(parakeet), thinks (llama.cpp), and speaks back (Supertonic TTS) sentence by
sentence. Interrupt it any time by speaking — barge-in stops playback.

The realtime pipeline runs over a WebSocket (`ws://<host>:7778/ws`, one port
above the HTTP port) using the same local STT/LLM/TTS services. Use headphones
to avoid speaker→mic echo. Settings (voice, language, steps, speed, system
prompt, API URL) are shared with the normal chat via the Settings panel.

```
Browser mic ──16kHz PCM──→ ws :7778 ──→ VAD → parakeet → llama.cpp → Supertonic
                          (binary)                                 ↓ sentence TTS
Browser speaker ←──PCM─────────────────────────────────────────────┘
```
