# Realtime Orb

A hands-free speech-to-speech voice assistant. Tap the orb, talk, it replies — everything runs on your machine.

```
Browser mic ──16kHz PCM──► ws :7778 ──► VAD → parakeet STT → llama.cpp LLM → Kokoro-82M TTS
                                              ↓ sentence by sentence
Browser speaker ◄──PCM────────────────────────────────────────────────┘
```

## Prerequisites

*   **Python 3.12** — Kokoro depends on PyTorch, which works on Python 3.12 on Intel Macs (Python 3.13+ lacks compatible wheels). Apple Silicon can use any Python >= 3.10.
*   **[llama.cpp](https://github.com/ggerganov/llama.cpp) server** running with a model (e.g. `llama-server -m model.gguf --port 8080`).
*   **parakeet.cpp binary** for speech-to-text (macOS download — see step 3).
*   Headphones strongly recommended — the assistant hears itself through your speakers otherwise.

## Install & run

### 1. Clone and create a virtual environment

```bash
git clone git@github.com:wprudencio/my-ai-chat-client.git
cd my-ai-chat-client
git checkout realtime-lite

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` pins `numpy<2` — PyTorch 2.2 on Intel Mac is compiled against NumPy 1.x so newer NumPy will crash.

### 2. Download the STT model

```bash
mkdir -p models
curl -L -o models/tdt_ctc-110m-f16.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf
```

### 3. Place the parakeet-server binary

Download the macOS binary from [parakeet.cpp releases](https://github.com/mudler/parakeet.cpp/releases), extract it, and put it in `bin/`:

```bash
mkdir -p bin
# move the downloaded binary:
mv ~/Downloads/parakeet-server bin/
chmod +x bin/parakeet-server
```

### 4. Start llama.cpp

In a separate terminal:

```bash
llama-server -m your-model.gguf --port 8080
```

### 5. Start the orb

```bash
./run.sh
```

Open **[http://localhost:7777](http://localhost:7777)** (or `http://0.0.0.0:7777` for LAN access).

On first launch Kokoro downloads its model (~300 MB). This happens once; subsequent starts are instant.

## Settings

Click the gear icon (top-right) to open the settings drawer. All fields auto-save to your browser's localStorage and push to the server.

*   **Voice** — 22 Kokoro voices across 8 languages
*   **Language** — switches the Kokoro model and the parakeet STT language
*   **Speed** — TTS playback speed 0.5× to 2.0×
*   **Max tokens** — LLM response length
*   **System prompt, LLM URL, API key, model, STT URL**

## Override defaults

```bash
PORT=9999 LLM_API=http://other-machine:8080/v1/chat/completions ./run.sh
```

## Structure

```
├── server.py          Flask web server + background TTS loader
├── realtime.py        WebSocket pipeline (VAD → STT → LLM → TTS)
├── static/
│   ├── app.js         WS client, mic capture, audio playback, settings
│   └── styles.css     orb UI, settings drawer, pulse rings
├── templates/
│   └── index.html     single-page orb + transcript + settings drawer
├── run.sh             starts parakeet STT + the Flask/WS server
├── requirements.txt
├── bin/               parakeet-server binary (gitignored)
├── models/            STT models (gitignored)
└── .venv/             Python 3.12 venv (gitignored)
```

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | `8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) server | `8080` |
| TTS | [Kokoro-82M](https://github.com/hexgrad/kokoro) | in-process |
| UI | Flask + vanilla JS | `7777` HTTP · `7778` WS |

## Troubleshooting

**"No module named kokoro" or import errors**
Run `source .venv/bin/activate && pip install -r requirements.txt` again — make sure the venv is active.

**Parakeet binary not found**
Ensure `bin/parakeet-server` exists and is executable (`chmod +x bin/parakeet-server`).

**STT model not found**
Download it: `mkdir -p models && curl -L -o models/tdt_ctc-110m-f16.gguf https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-f16.gguf`

**No audio from the assistant**
Check the browser console (F12) for "playback AudioContext" warnings. If the AudioContext is suspended, click the page once to allow audio, then tap the orb again.

**"LLM returned no content" toast**
llama.cpp isn't running or the API URL is wrong. Check Settings → LLM API URL and make sure llama-server is started.