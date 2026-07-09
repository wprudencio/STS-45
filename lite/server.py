#!/usr/bin/env python3
"""
Supertonic Realtime Lite — a hands-free, speech-to-speech voice assistant.

Just the realtime orb: mic -> VAD -> parakeet STT -> llama.cpp LLM
-> Supertonic TTS -> speaker, over a single WebSocket. No chat history,
no sidebar, no push-to-talk. Open the page, tap the orb, and talk.

    Browser mic ──16kHz PCM──► ws :PORT ──► VAD -> STT -> LLM -> TTS
    Browser speaker ◄──PCM────────────────────────────────────────┘
"""

import argparse
import sys
import threading

import numpy as np
from flask import Flask, render_template, request
from supertonic import TTS

try:
    import realtime as _realtime
except Exception as _rt_err:
    _realtime = None
    print(f"⚠️  Realtime module unavailable (voice mode disabled): {_rt_err}")

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
STT_API = "http://localhost:8081"

SYS_PROMPT = (
    "You are a friendly, helpful assistant. Respond in the same language as the user. "
    "Keep answers concise and natural for text-to-speech. "
    "Avoid markdown, lists, URLs, or special formatting. "
    "Use short to medium sentences. Avoid asterisks and emojis."
)

app = Flask(__name__)

tts = None
tts_lock = threading.Lock()

config = {
    "lang": "en",
    "voice": "M1",
    "steps": 5,
    "speed": 1.15,
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

# Realtime WebSocket port (HTTP port + 1 by default).
RT_WS_PORT = 7778


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_api_url=config["api_url"],
        default_stt_api_url=config["stt_api_url"],
        ws_port=RT_WS_PORT,
    )


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json() or {}
    for key in ("lang", "voice", "api_url", "stt_api_url", "model"):
        if key in data and data[key] is not None:
            config[key] = data[key]
    for key in ("steps", "speed"):
        if key in data and data[key] is not None:
            try:
                config[key] = type(config[key])(data[key])
            except (TypeError, ValueError):
                pass
    return {"status": "ok"}


@app.route("/api/health")
def api_health():
    return {
        "tts_ready": tts is not None,
        "realtime": _realtime is not None,
        "config": config,
    }


def _load_tts_background():
    global tts
    try:
        tts = TTS(auto_download=True)
        print("✅ TTS loaded")
    except Exception as e:
        print(f"⚠️ TTS loading failed (will retry on first use): {e}")


def main():
    global config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="Supertonic Realtime Lite")
    parser.add_argument("--host", default="0.0.0.0", help="Host (use 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=7777, help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=0, help="Realtime WebSocket port (default: HTTP port + 1)")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
    parser.add_argument("--stt-api", default=STT_API, help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="M1", help="Voice")
    parser.add_argument("--lang", default="en", help="Language")
    parser.add_argument("--steps", type=int, default=5, help="TTS diffusion steps")
    parser.add_argument("--speed", type=float, default=1.15, help="TTS speed")
    args = parser.parse_args()

    config.update({
        "lang": args.lang,
        "voice": args.voice,
        "steps": args.steps,
        "speed": args.speed,
        "api_url": args.api,
        "stt_api_url": args.stt_api,
        "model": args.model,
    })

    RT_WS_PORT = args.ws_port or (args.port + 1)

    print("🚀 Loading Supertonic TTS in background...")
    threading.Thread(target=_load_tts_background, daemon=True).start()

    if _realtime is not None:
        try:
            _realtime.start(args.host, RT_WS_PORT, sys.modules[__name__])
        except Exception as e:
            print(f"⚠️  Realtime server failed to start: {e}")
    else:
        print("⚠️  Realtime mode unavailable (websockets not installed).")

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 Supertonic Realtime Lite            ║
║   Open: http://{args.host}:{args.port}          ║
║   WS:   ws://{args.host}:{RT_WS_PORT}/ws           ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}              ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()