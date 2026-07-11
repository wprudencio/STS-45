# STS-45

Hands-free speech-to-speech voice assistant. Tap the orb, talk, it replies — everything runs on your machine.

```
Browser mic ──16kHz PCM──► ws :7778 ──► VAD → parakeet → llama.cpp → Piper TTS
                          (binary)                                 ↓ per clause
Browser speaker ◄──PCM─────────────────────────────────────────────┘
```

## Quick Start — Docker (one command)

The easiest way: `docker compose up` builds and starts everything — llama.cpp (LLM), parakeet.cpp (STT), and the STS-45 app (TTS+UI). Models download automatically on first launch.

```bash
git clone git@github.com:wprudencio/STS-45.git
cd STS-45
git checkout piper-tts
docker compose up
```

Then open **[http://localhost:7777](http://localhost:7777)**.

What each container does:

| Service | Image | Port | What it does |
|---------|-------|------|--------------|
| `llama-server` | ubuntu + [llama.app](https://llama.app) installer | `8080` | Downloads `LFM2.5-230M-Q4_K_M.gguf` (~150 MB), runs `llama server` |
| `parakeet-stt` | ubuntu + parakeet.cpp v0.4.0 binary | `8081` | Downloads `tdt_ctc-110m-f16.gguf`, runs `parakeet-server` |
| `app` | python:3.12-slim + piper-tts | `7777` (HTTP) · `7778` (WS) | The orb, WebSocket pipeline, Piper voices (downloaded on first use) |

Models are stored in named volumes (`llama-models`, `stt-models`, `piper-voices`) so they don't re-download on restarts. First launch will pull/build the images and fetch models — give it a few minutes.

### Override the LLM model

Edit `docker-compose.yml`:

```yaml
environment:
  MODEL_FILE: "LFM2.5-230M-Q5_K_M.gguf"  # higher quality, ~170 MB
  HF_REPO: "LiquidAI/LFM2.5-230M-GGUF"
```

Available quantizations: `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `BF16` (see [the model card](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/tree/main)).

### Rebuild only the app

```bash
docker compose up -d --build app
```

---

## Quick Start — local (no Docker)

If you want to run pieces on bare metal (e.g. the LLM on a GPU box, the orb on your laptop):

### Prerequisites

*   **Python 3.12** — Piper works on any Python ≥ 3.9, no torch needed.
*   **[llama.cpp](https://github.com/ggerganov/llama.cpp) server** — install via `curl -LsSf https://llama.app/install.sh | sh`, then `llama server -m model.gguf --port 8080`.
*   **parakeet-server** binary — download from [parakeet.cpp releases](https://github.com/mudler/parakeet.cpp/releases), put it in `bin/`.
*   Headphones recommended to avoid speaker→mic echo.

### Install & run

```bash
# 1. venv + deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-f16.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf

# 3. parakeet-server binary (one-time)
mkdir -p bin && chmod +x bin/parakeet-server   # place the binary here

# 4. llama.cpp (one-time)
curl -LsSf https://llama.app/install.sh | sh
# start it in another terminal:
llama server -m LFM2.5-230M-Q4_K_M.gguf --port 8080

# 5. start the orb
./run.sh
```

Open **[http://localhost:7777](http://localhost:7777)** (or `http://0.0.0.0:7777` for LAN).

Override defaults:

```bash
PORT=9999 LLM_API=http://other-host:8080/v1/chat/completions ./run.sh
```

## Structure

```
├── server.py          Flask web server + background Piper loader
├── realtime.py        WebSocket pipeline (VAD → STT → LLM → TTS)
├── static/
│   ├── app.js         WS client, mic capture, audio playback, settings
│   └── styles.css     orb UI, settings drawer, pulse rings
├── templates/
│   └── index.html     single-page orb + transcript + settings drawer
├── run.sh             starts parakeet STT + the Flask/WS server
├── requirements.txt
├── docker-compose.yml
├── docker/
│   ├── llama/Dockerfile       ubuntu + llama.app installer + LFM2.5-230M
│   ├── parakeet/Dockerfile    ubuntu + parakeet.cpp v0.4.0 binary
│   └── app/Dockerfile         python:3.12-slim + piper-tts
├── bin/               parakeet-server binary (gitignored, local only)
├── models/            STT + Piper voice models (gitignored)
└── .venv/             Python 3.12 venv (gitignored, local only)
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.4.0 | `8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) (via [llama.app](https://llama.app)) + `LiquidAI/LFM2.5-230M-GGUF` | `8080` |
| TTS | [Piper](https://github.com/rhasspy/piper) (ONNX Runtime) | in-process |
| UI | Flask + vanilla JS | `7777` HTTP · `7778` WS |

## Voices

15 Piper voices across 8 languages: Lessac, Amy, LibriTTS, Ryan (EN), Alan, Semaine (UK),
Faber (BR), Carl (ES), Siwis (FR), Thorsten (DE), JVS (JP), GlobalVoice (KO), Paola (IT).
Select from the Settings drawer (gear icon, top-right). New voices auto-download from
HuggingFace on first use and are cached in the `piper-voices` Docker volume (or `models/piper/` locally).

## Settings

Click the gear icon (top-right) to open the settings drawer. All fields auto-save to
`localStorage` and push to the server via `/api/settings`:

* **Voice** — Piper voice (determines language)
* **Language** — used by parakeet STT
* **Max tokens** — LLM response length
* **System prompt, LLM URL, API key, model, STT URL**

## Troubleshooting

**Docker: `Could not find model file`**
The wget fallback failed. Set `MODEL_FILE` in `docker-compose.yml` to a filename that exists in the `HF_REPO` (see [model files](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/tree/main)).

**Docker: app can't reach llama-server**
`depends_on` only waits for container start, not readiness. The first request may 503 for ~30 s while `llama-server` loads — the orb auto-retries.

**Local: "No module named piper"**
Run `source .venv/bin/activate && pip install -r requirements.txt` again — make sure the venv is active.

**Local: parakeet binary not found**
Ensure `bin/parakeet-server` exists and is executable (`chmod +x bin/parakeet-server`).

**Local: STT model not found**
`mkdir -p models && curl -L -o models/tdt_ctc-110m-f16.gguf https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf`

**No audio from the assistant**
Check the browser console (F12). If the AudioContext is suspended, click the page once to allow audio, then tap the orb again.

**"LLM returned no content" toast**
`llama-server` isn't running or the API URL is wrong. Check Settings → LLM API URL.