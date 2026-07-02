# Supertonic — Voice Chat

Local voice assistant with realtime WebSocket pipeline: mic → VAD → STT → LLM → TTS → speaker.

Two modes available:

## Realtime Voice (orb UI)

```
Browser mic ──→ WebSocket ──→ VAD ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
                              (RMS)      (STT)           (LLM)        (ONNX Runtime)
```

Continuous conversation: tap the orb, start talking. VAD detects speech automatically,
transcribes, generates a response, and speaks it back — all in real time.

## Classic Chat (PTT UI)

```
Browser mic ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
                (STT)           (LLM)         (ONNX Runtime)
```

Hold mic to talk, release to transcribe, Enter to send. Text-first chat UI.

## Quick Start

```bash
# 1. Download STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q5_k.gguf

# 2. Start llama.cpp on host
llama-server -m your-model.gguf --port 8080

# 3. Start everything else (realtime voice mode)
docker compose up -d

# 4. Open http://localhost:7777 — tap the orb to talk

# For the classic PTT + text chat UI:
# docker compose run --rm -p 7777:7777 chat python3 chat_ui.py --host 0.0.0.0 --port 7777
```

## Stack

| Role | Tech | Port |
|------|------|------|
| VAD | RMS-based (server-side) | — |
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (NVIDIA NeMo → ggml) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `:8080` |
| TTS | [Supertonic 3](https://github.com/supertone-inc/supertonic) (ONNX Runtime) | — |
| Server | FastAPI + Uvicorn (WebSocket + HTTP) | `:7777` |

## Architecture

### Realtime Mode (`realtime_server.py`)

The browser streams mic audio over WebSocket to the server. The server runs an
RMS-based VAD, transcribes speech via parakeet.cpp, sends the text to llama.cpp,
splits the response into sentences, and synthesizes each with Supertonic TTS.
TTS audio chunks stream back over WebSocket and play through the browser.

```
Browser                               Server
  │                                     │
  │── audio chunk (PCM 16kHz) ─────────→│
  │── audio chunk ─────────────────────→│
  │── ... ─────────────────────────────→│
  │                                     │ VAD detects speech end
  │                                     │ STT → parakeet.cpp
  │                                     │ LLM → llama.cpp (streaming)
  │←── status: processing ──────────── │
  │                                     │ TTS → Supertonic (per sentence)
  │←── audio chunk (WAV) ───────────── │
  │←── audio chunk (WAV) ───────────── │
  │←── status: idle ───────────────── │
```

### Classic Mode (`chat_ui.py`)

The browser records audio on push-to-talk, sends the full recording to `/api/stt`,
receives text, sends it to `/api/chat` for SSE streaming from the LLM, and
optionally plays TTS via `/api/tts`.
