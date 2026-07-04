#!/usr/bin/env python3
"""
Supertonic Voice Chat — Web UI com microfone + teclado + LLM + TTS local.

The UI lives in templates/index.html + static/{styles.css,app.js}; this file
only holds the Flask API surface (chat, STT, TTS) and the realtime bridge.
"""

import argparse
import json
import sys
import threading
import time
import base64
import io
import wave

import numpy as np
import requests
from flask import Flask, Response, request, render_template
from supertonic import TTS

try:
    import realtime as _realtime
except Exception as _rt_err:  # websockets missing -> core chat still works
    _realtime = None
    print(f"⚠️  Realtime module unavailable (voice mode disabled): {_rt_err}")

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
STT_API = "http://localhost:8081"

# Realtime WebSocket port (set in main(); one above the HTTP port by default).
RT_WS_PORT = 7778

SYS_PROMPT = (
    "You are a friendly, helpful assistant. Respond in the same language as the user. "
    "Keep answers concise and natural for text-to-speech. "
    "Avoid markdown, lists, URLs, or special formatting. "
    "Use short to medium sentences. Avoid asterisks and emojis."
)

app = Flask(__name__)

# Estado global (carregado no startup)
tts = None
style = None
tts_lock = threading.Lock()

# Configurações
config = {
    "lang": "en",
    "voice": "M1",
    "steps": 5,
    "speed": 1.15,
    "api_url": LLAMA_API,
    "stt_api_url": STT_API,
    "model": "default",
}

# Histórico do LLM
messages = [{"role": "system", "content": SYS_PROMPT}]


# --- Index page (UI is in templates/index.html + static/) ---
@app.route("/")
def index():
    return render_template(
        "index.html",
        default_api_url=config["api_url"],
        default_stt_api_url=config["stt_api_url"],
        ws_port=RT_WS_PORT,
    )


# --- Streaming chat endpoint ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    global messages, style

    data = request.get_json()
    user_msg = (data.get("message") or "").strip()

    # Limpa histórico se pedir
    if user_msg.lower() in ("clear", "/clear"):
        messages = [{"role": "system", "content": SYS_PROMPT}]
        return {"status": "cleared", "message": "History cleared"}

    # Atualiza config se veio do frontend
    for key in ("lang", "voice", "steps", "speed"):
        if key in data:
            config[key] = data[key]
    # Aceita API URL, key, e system prompt do frontend
    api_url = data.get("api_url", config["api_url"]).strip()
    api_key = data.get("api_key", "").strip()
    sys_prompt = data.get("sys_prompt", "").strip()
    if sys_prompt:
        messages[0] = {"role": "system", "content": sys_prompt}
    else:
        messages[0] = {"role": "system", "content": SYS_PROMPT}
    if not api_url:
        api_url = config["api_url"]
    if "voice" in data and tts is not None:
        try:
            style = tts.get_voice_style(voice_name=data["voice"])
        except Exception:
            pass

    # --- History control for edit / regenerate / branch ---------------------
    # If `history` is supplied, it replaces the conversation (minus the system
    # prompt). `append` (default True) controls whether `message` is appended as
    # a fresh user turn; `regenerate` replaces the trailing assistant turn.
    history = data.get("history")
    if isinstance(history, list):
        messages = [messages[0]] + [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
    append_user = data.get("append", True)
    is_regen = bool(data.get("regenerate", False))
    if is_regen:
        append_user = False
    temperature = float(data.get("temperature", 0.7))

    def generate():
        global messages

        if append_user and user_msg:
            messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": int(data.get("max_tokens", 2048)),
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.time()
        t_first = None
        try:
            r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
            r.encoding = 'utf-8'
        except Exception as e:
            # Some servers reject `stream_options`; retry once without it.
            if payload.pop("stream_options", None):
                try:
                    r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
                    r.raise_for_status()
                    r.encoding = 'utf-8'
                except Exception as e2:
                    yield f"data: {json.dumps({'type': 'error', 'text': str(e2)})}\n\n"
                    return
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
                return

        content_parts = []
        has_content = False
        usage = None

        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                content = delta.get("content")

                if reasoning and not has_content:
                    if t_first is None:
                        t_first = time.time()
                    yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
                elif content:
                    if t_first is None:
                        t_first = time.time()
                    has_content = True
                    content_parts.append(content)
                    yield f"data: {json.dumps({'type': 'text', 'text': content})}\n\n"
        except Exception:
            # Client disconnected (stop pressed) or upstream error.
            try:
                r.close()
            except Exception:
                pass
            return
        finally:
            try:
                r.close()
            except Exception:
                pass

        full = "".join(content_parts)
        if is_regen:
            if messages and messages[-1].get("role") == "assistant":
                messages[-1] = {"role": "assistant", "content": full}
            else:
                messages.append({"role": "assistant", "content": full})
        else:
            messages.append({"role": "assistant", "content": full})

        t_end = time.time()
        total_ms = int((t_end - t0) * 1000)
        ttft_ms = int((t_first - t0) * 1000) if t_first else 0
        tokens = None
        if usage:
            tokens = usage.get("completion_tokens") or usage.get("total_tokens")
        if not tokens:
            tokens = max(1, len(full) // 4)  # rough estimate (~4 chars/token)
        gen_secs = (t_end - t_first) if t_first else (t_end - t0)
        tps = (tokens / gen_secs) if gen_secs > 0 else 0

        yield f"data: {json.dumps({'type': 'stats', 'ttft_ms': ttft_ms, 'total_ms': total_ms, 'tokens': int(tokens), 'tps': round(tps, 1)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# --- Load conversation history ---
@app.route("/api/chat/load", methods=["POST"])
def api_chat_load():
    global messages
    data = request.get_json()
    msgs = data.get("messages", [])
    sys_prompt = data.get("sys_prompt", "").strip()
    base = [{"role": "system", "content": sys_prompt or SYS_PROMPT}]
    messages = base + [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return {"status": "ok", "count": len(messages)}


# --- Standalone TTS endpoint (for play-message feature) ---
@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json()
    text = (data.get("text", "")).strip()
    if not text:
        return {"error": "empty text"}, 400
    lang = data.get("lang", config["lang"])
    voice = data.get("voice", config["voice"])
    steps = int(data.get("steps", config["steps"]))
    speed = float(data.get("speed", config["speed"]))
    if tts is None:
        return {"error": "TTS still loading (model download in progress)"}, 503
    try:
        vs = tts.get_voice_style(voice_name=voice)
        with tts_lock:
            wav, _dur = tts.synthesize(text=text, lang=lang, voice_style=vs, total_steps=steps, speed=speed)
        b64 = wav_to_base64(wav, tts.sample_rate)
        return {"audio": b64}
    except Exception as e:
        return {"error": str(e)}, 500


# --- STT endpoint (proxies to parakeet.cpp server) ---
@app.route("/api/stt", methods=["POST"])
def api_stt():
    if "file" not in request.files:
        return {"error": "no file"}, 400

    file = request.files["file"]
    lang = request.form.get("lang", "en")
    stt_api_url = request.form.get("stt_api", config.get("stt_api_url", STT_API)).strip()

    lang_map = {
        "en": "en", "pt": "pt", "es": "es", "fr": "fr",
        "de": "de", "ja": "ja", "ko": "ko",
    }
    parakeet_lang = lang_map.get(lang, "en")

    try:
        url = stt_api_url.rstrip("/") + "/v1/audio/transcriptions"
        resp = requests.post(
            url,
            files={"file": (file.filename or "audio.wav", file.read(), "audio/wav")},
            data={"language": parakeet_lang, "response_format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        return {"text": resp.json().get("text", "")}
    except Exception as e:
        return {"error": str(e)}, 500


def wav_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    """Converte numpy WAV para base64 (WAV format)."""
    wav_mono = wav.squeeze()
    wav_int16 = (np.clip(wav_mono, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _load_tts_background():
    global tts, style, config
    try:
        tts = TTS(auto_download=True)
        style = tts.get_voice_style(voice_name=config["voice"])
        print("✅ TTS loaded")
    except Exception as e:
        print(f"⚠️ TTS loading failed (will retry on first use): {e}")


def main():
    global tts, style, config, RT_WS_PORT

    parser = argparse.ArgumentParser(description="Supertonic Voice Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7777, help="Port")
    parser.add_argument("--ws-port", type=int, default=0, help="Realtime WebSocket port (default: HTTP port + 1)")
    parser.add_argument("--api", default=LLAMA_API, help="LLM API URL")
    parser.add_argument("--stt-api", default=STT_API, help="Parakeet STT server URL")
    parser.add_argument("--model", default="default", help="Model name")
    parser.add_argument("--voice", default="M1", help="Voice")
    parser.add_argument("--lang", default="en", help="Language")
    parser.add_argument("--steps", type=int, default=5, help="TTS steps")
    parser.add_argument("--speed", type=float, default=1.15, help="Speed")
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
    t = threading.Thread(target=_load_tts_background, daemon=True)
    t.start()

    if _realtime is not None:
        try:
            _realtime.start(args.host, RT_WS_PORT, sys.modules[__name__])
        except Exception as e:
            print(f"⚠️  Realtime server failed to start: {e}")
    else:
        print("⚠️  Realtime mode unavailable (websockets not installed).")

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 Supertonic Voice Chat              ║
║   Open: http://{args.host}:{args.port}          ║
║   WS:   ws://{args.host}:{RT_WS_PORT}/ws           ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}      ║
║   Steps: {args.steps}  |  Speed: {args.speed}          ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
