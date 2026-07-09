#!/usr/bin/env python3
"""
Realtime voice orb — VAD → parakeet STT → llama.cpp LLM → Kokoro TTS → speaker.

    Browser mic ──16kHz PCM──► ws :PORT ──► VAD -> STT -> LLM -> TTS
    Browser speaker ◄──PCM────────────────────────────────────────┘

Open http://localhost:PORT, tap the orb, and talk.
"""

import argparse
import sys
import threading

from flask import Flask, render_template, request
from kokoro import KPipeline

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

tts = {}
tts_lock = threading.Lock()

config = {
    "lang": "en",
    "voice": "af_heart",
    "speed": 1.0,
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

RT_WS_PORT = 7778

# Language code mapping: iso → kokoro lang_code
LANG_MAP = {
    "en": "a", "pt": "p", "es": "e", "fr": "f",
    "de": "a", "ja": "j", "ko": "a", "it": "i",
}


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
    for key in ("lang", "voice", "speed", "api_url", "stt_api_url", "model"):
        if key in data and data[key] is not None:
            try:
                config[key] = type(config[key])(data[key])
            except (TypeError, ValueError):
                config[key] = data[key]
    return {"status": "ok"}


@app.route("/api/health")
def api_health():
    return {
        "tts_ready": bool(tts),
        "realtime": _realtime is not None,
        "config": config,
    }


def _load_tts_background():
    global tts
    try:
        tts["a"] = KPipeline(lang_code="a")
        print("✅ Kokoro TTS loaded")
    except Exception as e:
        print(f"⚠️ Kokoro loading failed (will retry on first use): {e}")


def main():
    global config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="Realtime voice orb (Kokoro TTS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=7777, help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=0, help="WS port (default: HTTP port + 1)")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
    parser.add_argument("--stt-api", default=STT_API, help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice")
    parser.add_argument("--lang", default="en", help="Language")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed")
    args = parser.parse_args()

    config.update({
        "lang": args.lang,
        "voice": args.voice,
        "speed": args.speed,
        "api_url": args.api,
        "stt_api_url": args.stt_api,
        "model": args.model,
    })

    RT_WS_PORT = args.ws_port or (args.port + 1)

    print("🚀 Loading Kokoro TTS in background...")
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
║   🎤 Realtime Orb (Kokoro TTS)           ║
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