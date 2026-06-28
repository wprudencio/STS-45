#!/usr/bin/env python3
"""
Supertonic Voice Chat — Web UI com microfone + teclado + LLM + TTS local.
"""

import argparse
import json
import re
import sys
import threading
import time
import base64
import io
import wave
from typing import Optional

import numpy as np
import requests
from flask import Flask, Response, request, render_template_string
from supertonic import TTS

LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
STT_API = "http://localhost:8080"

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

# --- Sentence Buffer ---
class SentenceBuffer:
    def __init__(self):
        self.buffer = ""

    def add(self, token: str) -> Optional[str]:
        self.buffer += token
        if len(self.buffer) > 180:
            s = self.buffer.strip()
            self.buffer = ""
            return s
        if re.search(r"[.!?…]\s*$", self.buffer):
            if not re.search(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc)\.$", self.buffer):
                s = self.buffer.strip()
                self.buffer = ""
                return s
        return None

    def flush(self) -> Optional[str]:
        if self.buffer.strip():
            s = self.buffer.strip()
            self.buffer = ""
            return s
        return None


# --- Streaming chat endpoint ---
@app.route("/api/chat", methods=["POST"])
def api_chat():
    global messages

    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return {"error": "empty message"}, 400

    # Limpa histórico se pedir
    if user_msg.lower() in ("clear", "/clear"):
        messages = [{"role": "system", "content": SYS_PROMPT}]
        return {"status": "cleared", "message": "History cleared"}

    # Atualiza config se veio do frontend
    for key in ("lang", "voice", "steps", "speed"):
        if key in data:
            config[key] = data[key]
    # Aceita API URL e key do frontend (sobrescreve defaults locais)
    api_url = data.get("api_url", config["api_url"]).strip()
    api_key = data.get("api_key", "").strip()
    if not api_url:
        api_url = config["api_url"]
    if "voice" in data:
        try:
            global style
            style = tts.get_voice_style(voice_name=data["voice"])
        except:
            pass

    def generate():
        global messages
        buf = SentenceBuffer()

        # Stream do LLM
        messages.append({"role": "user", "content": user_msg})
        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.7,
        }

        # Headers com auth se tiver API key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            r = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        content_parts = []
        has_content = False

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                d = line[6:]
                if d.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                content = delta.get("content")

                if reasoning and not has_content:
                    yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
                elif content:
                    has_content = True
                    content_parts.append(content)
                    yield f"data: {json.dumps({'type': 'text', 'text': content})}\n\n"

                    # Detecta sentença completa → sintetiza TTS
                    sentence = buf.add(content)
                    if sentence is not None and len(sentence) > 3:
                        try:
                            with tts_lock:
                                wav, dur = tts.synthesize(
                                    text=sentence,
                                    lang=config["lang"],
                                    voice_style=style,
                                    total_steps=config["steps"],
                                    speed=config["speed"],
                                )
                            b64 = wav_to_base64(wav, tts.sample_rate)
                            yield f"data: {json.dumps({'type': 'audio', 'data': b64})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'text': f'TTS error: {e}'})}\n\n"

        # Flush final
        sentence = buf.flush()
        if sentence and len(sentence) > 3:
            try:
                with tts_lock:
                    wav, dur = tts.synthesize(
                        text=sentence,
                        lang=config["lang"],
                        voice_style=style,
                        total_steps=config["steps"],
                        speed=config["speed"],
                    )
                b64 = wav_to_base64(wav, tts.sample_rate)
                yield f"data: {json.dumps({'type': 'audio', 'data': b64})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'text': f'TTS error: {e}'})}\n\n"

        full = "".join(content_parts)
        messages.append({"role": "assistant", "content": full})

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


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


# --- HTML UI ---
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Supertonic — Voice Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --black: #000000;
      --deep: #020202;
      --charcoal: #303030;
      --char2: #323131;
      --mid: #545352;
      --light: #737270;
      --bg: #eeeeee;
      --bg2: #f6f6f6;
      --orange: #ea6626;
      --soft: #e8ab8f;
      --white: #ffffff;

      --sp1: 10px;
      --sp2: 20px;
      --sp3: 32px;
      --sp4: 48px;
      --font: 'Geist Mono', monospace;
      --mono: 'Geist Mono', monospace;
      --dur: 200ms;
      --ease: cubic-bezier(0.25, 0, 0, 1);
    }

    html, body { height: 100%; overflow: hidden; }

    body {
      background: var(--bg);
      color: var(--charcoal);
      font-family: var(--font);
      font-weight: 400;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      display: flex;
      flex-direction: column;
    }

    /* TOPBAR */
    .topbar {
      height: 64px;
      flex-shrink: 0;
      background: var(--deep);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 var(--sp4);
      border-bottom: 1px solid #181818;
      z-index: 50;
    }

    .brand { display: flex; align-items: center; gap: 12px; }

    .brand-mark {
      width: 24px;
      height: 24px;
      border: 1.5px solid var(--orange);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      padding: 3px;
    }
    .brand-mark span { background: var(--orange); display: block; }
    .brand-mark span:nth-child(2) { background: transparent; border: 1px solid var(--orange); }

    .brand-name {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--bg);
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .topbar-center { display: flex; align-items: center; gap: 8px; }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--light);
      transition: background var(--dur);
    }
    .status-dot.active {
      background: var(--orange);
      animation: pulse 2.5s ease-in-out infinite;
    }
    .status-dot.rec { background: var(--soft); animation: pulse 0.8s ease-in-out infinite; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    .status-text {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--light);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .topbar-right { display: flex; align-items: center; gap: var(--sp3); }

    .live-badge {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--light);
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .live-badge span { color: var(--soft); }

    /* SHELL */
    .shell {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 320px 1fr 260px;
      overflow: hidden;
    }

    /* PANEL LEFT */
    .panel-left {
      background: var(--bg2);
      border-right: 1px solid #d8d8d8;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }

    .panel-section {
      padding: var(--sp3);
      border-bottom: 1px solid #e0e0e0;
    }

    .section-label {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--light);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: var(--sp2);
    }

    .select-wrap { position: relative; margin-bottom: var(--sp2); }
    .select-wrap::after {
      content: '';
      position: absolute;
      right: 12px;
      top: 50%;
      width: 8px;
      height: 8px;
      border-right: 1.5px solid var(--light);
      border-bottom: 1.5px solid var(--light);
      transform: translateY(-70%) rotate(45deg);
      pointer-events: none;
    }

    .model-select {
      width: 100%;
      background: var(--white);
      border: 1px solid #d4d4d4;
      padding: 8px 28px 8px 10px;
      font-family: var(--mono);
      font-size: 13px;
      color: var(--charcoal);
      outline: none;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
    }
    .model-select:focus { border-color: var(--charcoal); }
    .model-select option { background: var(--white); color: var(--charcoal); }

    .sys-prompt {
      width: 100%;
      background: var(--white);
      border: 1px solid #d4d4d4;
      padding: 8px 10px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--charcoal);
      resize: none;
      outline: none;
      line-height: 1.55;
      min-height: 72px;
      transition: border-color var(--dur);
    }
    .sys-prompt:focus { border-color: var(--charcoal); }

    .slider-row { display: flex; flex-direction: column; gap: 5px; margin-bottom: var(--sp2); }
    .slider-row:last-child { margin-bottom: 0; }
    .slider-label { display: flex; justify-content: space-between; align-items: baseline; }
    .slider-name {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .slider-val {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--char2);
      font-weight: 500;
    }

    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      height: 2px;
      background: linear-gradient(to right, var(--charcoal) var(--fill, 0%), #d8d8d8 var(--fill, 0%));
      outline: none;
      cursor: pointer;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 10px;
      height: 10px;
      background: var(--orange);
      cursor: pointer;
      border-radius: 0;
    }
    input[type=range]::-moz-range-thumb {
      width: 10px;
      height: 10px;
      background: var(--orange);
      cursor: pointer;
      border: none;
      border-radius: 0;
    }

    .spec-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 5px 0;
      border-bottom: 1px solid #ebebeb;
    }
    .spec-row:last-child { border-bottom: none; }
    .spec-key {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      letter-spacing: 0.05em;
    }
    .spec-val {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--char2);
    }
    .spec-val.hl { color: var(--orange); }

    .clear-btn {
      width: 100%;
      background: transparent;
      border: 1px solid #d4d4d4;
      color: var(--mid);
      padding: 8px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      transition: border-color var(--dur), color var(--dur);
    }
    .clear-btn:hover { border-color: var(--charcoal); color: var(--charcoal); }

    /* PANEL CENTER */
    .panel-center {
      display: flex;
      flex-direction: column;
      background: var(--bg);
      border-right: 1px solid #d8d8d8;
      min-height: 0;
    }

    .chat-header {
      height: 60px;
      flex-shrink: 0;
      border-bottom: 1px solid #d8d8d8;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 var(--sp3);
    }
    .chat-header-title {
      font-size: 15px;
      font-weight: 500;
      color: var(--charcoal);
      letter-spacing: -0.01em;
    }
    .chat-header-meta {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--light);
      letter-spacing: 0.05em;
    }

    /* Init overlay */
    .init-overlay {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: var(--sp3);
      padding: var(--sp4);
    }
    .init-overlay.hidden { display: none; }

    .init-glyph {
      display: grid;
      grid-template-columns: repeat(4, 16px);
      gap: 3px;
      margin-bottom: var(--sp2);
    }
    .init-cell { width: 16px; height: 16px; background: #e0e0e0; }
    .init-cell:nth-child(3n) { background: var(--charcoal); }
    .init-cell:nth-child(7) { background: var(--orange); animation: cell-pulse 1.8s ease-in-out infinite; }

    @keyframes cell-pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    .init-title {
      font-size: 28px;
      font-weight: 500;
      color: var(--charcoal);
      letter-spacing: -0.02em;
      text-align: center;
      line-height: 1.2;
    }
    .init-sub {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--light);
      text-align: center;
      line-height: 1.7;
      letter-spacing: 0.02em;
    }

    /* Messages */
    .messages {
      flex: 1;
      overflow-y: auto;
      min-height: 0;
      padding: var(--sp3) var(--sp4);
      display: flex;
      flex-direction: column;
      gap: var(--sp3);
    }
    .messages.hidden { display: none; }

    .messages::-webkit-scrollbar { width: 3px; }
    .messages::-webkit-scrollbar-thumb { background: #ccc; }

    .message {
      display: flex;
      gap: var(--sp2);
      animation: msg-in 180ms var(--ease);
    }
    @keyframes msg-in {
      from { opacity: 0; transform: translateY(4px) }
      to { opacity: 1; transform: none }
    }
    .message.user { flex-direction: row-reverse; }

    .msg-avatar {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.05em;
      color: var(--light);
      padding-top: 20px;
      flex-shrink: 0;
      width: 36px;
      text-align: center;
    }
    .message.user .msg-avatar { color: var(--orange); }
    .message.reasoning .msg-avatar { color: var(--soft); }

    .msg-body { display: flex; flex-direction: column; gap: 3px; max-width: 78%; }
    .message.user .msg-body { align-items: flex-end; }

    .msg-header { display: flex; gap: 10px; align-items: baseline; }
    .message.user .msg-header { flex-direction: row-reverse; }

    .msg-name {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--light);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .message.user .msg-name { color: var(--orange); }
    .message.reasoning .msg-name { color: var(--soft); }

    .msg-time {
      font-family: var(--mono);
      font-size: 11px;
      color: #bbb;
      letter-spacing: 0.03em;
    }

    .msg-content {
      font-size: 16px;
      line-height: 1.65;
      color: var(--char2);
      word-break: break-word;
    }
    .msg-content p { margin-bottom: 0.75em; }
    .msg-content p:last-child { margin-bottom: 0; }
    .msg-content code {
      font-family: var(--mono);
      background: var(--bg2);
      padding: 2px 4px;
      border-radius: 3px;
      font-size: 14px;
    }
    .message.reasoning .msg-content {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--soft);
      font-style: italic;
      letter-spacing: 0.02em;
      padding: 6px 0;
    }
    .message.reasoning .msg-content::before {
      content: '// REASONING';
      display: block;
      font-style: normal;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.18em;
      color: var(--soft);
      margin-bottom: 4px;
      opacity: 0.7;
    }
    .message.user .msg-content {
      background: var(--white);
      border: 1px solid #e0e0e0;
      padding: 10px 14px;
      color: var(--charcoal);
      white-space: pre-wrap;
    }

    .typing-cursor {
      display: inline-block;
      width: 2px;
      height: 14px;
      background: var(--orange);
      margin-left: 2px;
      vertical-align: middle;
      animation: blink 1s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* Input */
    .input-area {
      border-top: 1px solid #d8d8d8;
      background: var(--bg2);
      padding: var(--sp2) var(--sp3);
      flex-shrink: 0;
    }
    .input-row { display: flex; gap: 8px; align-items: stretch; }

    .ptt-btn {
      background: transparent;
      border: 1px solid #d4d4d4;
      color: var(--charcoal);
      width: 54px;
      flex-shrink: 0;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all var(--dur) var(--ease);
    }
    .ptt-btn:hover { border-color: var(--charcoal); color: var(--orange); }
    .ptt-btn.recording {
      background: var(--soft);
      border-color: var(--soft);
      color: var(--white);
      animation: pttPulse 1s ease-in-out infinite;
    }
    .ptt-btn.recording:hover { color: var(--white); }
    @keyframes pttPulse {
      0%,100% { box-shadow: 0 0 0 0 rgba(232,171,143,0.5); }
      50% { box-shadow: 0 0 0 6px rgba(232,171,143,0); }
    }

    .user-input {
      flex: 1;
      background: var(--white);
      border: 1px solid #d4d4d4;
      padding: 10px var(--sp2);
      font-family: var(--font);
      font-size: 15px;
      color: var(--charcoal);
      resize: none;
      outline: none;
      line-height: 1.5;
      min-height: 54px;
      max-height: 180px;
      overflow-y: auto;
      transition: border-color var(--dur);
    }
    .user-input::placeholder { color: #aaa; }
    .user-input:focus { border-color: var(--charcoal); }

    .send-btn {
      background: var(--charcoal);
      color: var(--bg);
      border: none;
      height: 54px;
      padding: 0 18px;
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background var(--dur) var(--ease);
      white-space: nowrap;
      flex-shrink: 0;
    }
    .send-btn:hover { background: var(--black); }
    .send-btn:disabled { background: var(--mid); cursor: not-allowed; }

    .input-hint {
      margin-top: 5px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      letter-spacing: 0.04em;
    }

    /* PANEL RIGHT */
    .panel-right {
      background: var(--bg2);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }

    .metric-block {
      padding: var(--sp3);
      border-bottom: 1px solid #e0e0e0;
      display: flex;
      flex-direction: column;
      gap: var(--sp2);
    }
    .metric-row { display: flex; flex-direction: column; gap: 4px; }
    .metric-label { display: flex; justify-content: space-between; align-items: baseline; }
    .metric-name {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .metric-value {
      font-family: var(--mono);
      font-size: 14px;
      color: var(--char2);
      font-weight: 500;
    }
    .metric-value.active { color: var(--orange); }

    .bar-track { height: 2px; background: #d8d8d8; }
    .bar-fill { height: 100%; background: var(--light); transition: width 500ms var(--ease); }
    .bar-fill.hl { background: var(--orange); }

    .log-wrap {
      padding: 0 var(--sp3) var(--sp3);
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .log-stream {
      flex: 1;
      overflow-y: auto;
      background: var(--white);
      border: 1px solid #e4e4e4;
      padding: 8px;
      min-height: 100px;
      max-height: 280px;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }
    .log-stream::-webkit-scrollbar { width: 2px; }
    .log-stream::-webkit-scrollbar-thumb { background: #ccc; }
    .log-entry {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      line-height: 1.7;
      letter-spacing: 0.02em;
      animation: log-in 150ms var(--ease);
    }
    @keyframes log-in { from {opacity:0; transform:translateX(-3px)} to {opacity:1; transform:none} }
    .log-entry .ts { color: #c0c0c0; margin-right: 5px; }
    .log-entry.ok { color: #7a9e7e; }
    .log-entry.warn { color: var(--soft); }
    .log-entry.hl { color: var(--orange); }

    /* FOOTER */
    .footer {
      height: 48px;
      flex-shrink: 0;
      background: var(--deep);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 var(--sp4);
      border-top: 1px solid #111;
    }
    .footer-headline {
      font-size: 14px;
      font-weight: 500;
      color: var(--bg);
      letter-spacing: -0.01em;
      line-height: 1.15;
    }
    .footer-headline em { font-style: normal; color: var(--orange); }

    ::-webkit-scrollbar { width: 3px; height: 3px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #ccc; }

    /* Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.4);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .modal-overlay.hidden { display: none; }
    .modal {
      background: var(--bg2);
      border: 1px solid #d4d4d4;
      width: 560px;
      max-width: 95%;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .modal-header {
      padding: var(--sp3);
      border-bottom: 1px solid #d4d4d4;
      background: var(--white);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-title {
      font-size: 15px;
      font-weight: 500;
      color: var(--charcoal);
    }
    .modal-body { padding: var(--sp3); display: flex; flex-direction: column; gap: var(--sp2); }
    .modal-footer {
      padding: var(--sp3);
      border-top: 1px solid #d4d4d4;
      display: flex;
      justify-content: flex-end;
      gap: var(--sp2);
      background: var(--bg);
    }
    .input-lbl {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--light);
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .field-input {
      width: 100%;
      background: var(--white);
      border: 1px solid #d4d4d4;
      padding: 8px 10px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--charcoal);
      outline: none;
      transition: border-color var(--dur);
    }
    .field-input:focus { border-color: var(--charcoal); }
    .field-input::placeholder { color: #aaa; }

    /* Responsive */
    @media (max-width: 1100px) {
      .shell { grid-template-columns: 280px 1fr 240px; }
    }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
      .panel-left, .panel-right { max-height: 200px; }
      .panel-left { border-right: none; border-bottom: 1px solid #d8d8d8; }
      .panel-right { border-left: none; border-top: 1px solid #d8d8d8; }
      .topbar-right { display: none; }
    }
  </style>
</head>
<body>

  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">
        <span></span><span></span><span></span><span></span>
      </div>
      <span class="brand-name">Supertonic &middot; Voice Chat</span>
    </div>
    <div class="topbar-center">
      <div class="status-dot" id="statusDot"></div>
      <span class="status-text" id="statusText">Ready</span>
    </div>
    <div class="topbar-right">
      <span class="live-badge">Voice: <span id="liveVoice">M1</span></span>
      <span class="live-badge">Lang: <span id="liveLang">EN</span></span>
    </div>
  </header>

  <div class="shell">

    <aside class="panel-left">
      <div class="panel-section">
        <div class="section-label">Voice</div>
        <div class="select-wrap">
          <select class="model-select" id="voice" onchange="onVoiceChange()">
            <option value="M1">M1 &middot; Male deep</option>
            <option value="M2">M2 &middot; Male mid</option>
            <option value="M3">M3 &middot; Male bright</option>
            <option value="M4">M4 &middot; Male warm</option>
            <option value="M5">M5 &middot; Male neutral</option>
            <option value="F1">F1 &middot; Female deep</option>
            <option value="F2">F2 &middot; Female mid</option>
            <option value="F3">F3 &middot; Female bright</option>
            <option value="F4">F4 &middot; Female warm</option>
            <option value="F5">F5 &middot; Female neutral</option>
          </select>
        </div>
        <div class="section-label" style="margin-top: var(--sp2)">Language</div>
        <div class="select-wrap">
          <select class="model-select" id="lang" onchange="onLangChange()">
            <option value="en">EN &middot; English</option>
            <option value="pt">PT &middot; Portuguese</option>
            <option value="es">ES &middot; Spanish</option>
            <option value="fr">FR &middot; French</option>
            <option value="de">DE &middot; German</option>
            <option value="ja">JP &middot; Japanese</option>
            <option value="ko">KR &middot; Korean</option>
          </select>
        </div>
      </div>

      <div class="panel-section">
        <div class="section-label">TTS Parameters</div>
        <div class="slider-row">
          <div class="slider-label">
            <span class="slider-name">Diffusion Steps</span>
            <span class="slider-val" id="stepsVal">5</span>
          </div>
          <input type="range" id="steps" min="2" max="12" step="1" value="5"
            oninput="document.getElementById('stepsVal').textContent=this.value; updateSlider(this)">
        </div>
        <div class="slider-row">
          <div class="slider-label">
            <span class="slider-name">Speed</span>
            <span class="slider-val" id="speedVal">1.15</span>
          </div>
          <input type="range" id="speed" min="0.7" max="2.0" step="0.05" value="1.15"
            oninput="document.getElementById('speedVal').textContent=parseFloat(this.value).toFixed(2); updateSlider(this)">
        </div>
      </div>

      <div class="panel-section">
        <div class="section-label">System Prompt</div>
        <textarea class="sys-prompt" id="sysPrompt" rows="4">You are a friendly, helpful assistant. Respond in the same language as the user. Keep answers concise and natural for text-to-speech. Avoid markdown, lists, URLs, or special formatting. Use short to medium sentences. Avoid asterisks and emojis.</textarea>
      </div>

      <div class="panel-section">
        <div class="section-label" style="display:flex; justify-content:space-between;">
          <span>LLM Connection</span>
          <span style="cursor:pointer; color:var(--orange); font-size:10px" onclick="openModal()">Configure</span>
        </div>
      </div>

      <div class="panel-section">
        <button class="clear-btn" onclick="clearChat()">Clear Conversation</button>
      </div>
    </aside>

    <main class="panel-center">
      <div class="chat-header">
        <span class="chat-header-title">Voice Session</span>
        <span class="chat-header-meta" id="sessionMeta">SES &middot; 00:00</span>
      </div>

      <div class="init-overlay" id="initOverlay">
        <div class="init-glyph">
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
          <div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div><div class="init-cell"></div>
        </div>
        <div class="init-title">Hold mic to talk<br>or type a message.</div>
        <div class="init-sub">
          TTS runs locally on your device.<br>
          Streaming synthesis, sentence by sentence.<br>
          No audio data leaves your machine.
        </div>
      </div>

      <div class="messages hidden" id="messages"></div>

      <div class="input-area">
        <div class="input-row">
          <button class="ptt-btn" id="pttBtn" title="Hold to talk (or hold Space)" aria-label="Push to talk">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
          </button>
          <textarea class="user-input" id="userInput" placeholder="Type a message..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this); onInputChange()"></textarea>
          <button class="send-btn" id="sendBtn" onclick="sendText()" disabled>
            Send
            <svg viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" width="10" height="10">
              <path d="M1 5h8M5 1l4 4-4 4" />
            </svg>
          </button>
        </div>
        <div class="input-hint">Hold Mic or Space to dictate (appends) &middot; &crarr; Send &middot; Shift+&crarr; New line</div>
      </div>
    </main>

    <aside class="panel-right">

      <div class="metric-block" style="border-bottom:none; padding-bottom:var(--sp2)">
        <div class="section-label" style="margin-bottom:0">Event Log</div>
      </div>
      <div class="log-wrap">
        <div class="log-stream" id="logStream"></div>
      </div>
    </aside>
  </div>

  <footer class="footer">
    <div class="footer-headline">
      Hold to talk, release to send.<br>
      <em>On-device</em> text-to-speech synthesis.
    </div>
  </footer>

  <div id="settingsModal" class="modal-overlay hidden">
    <div class="modal">
      <div class="modal-header">
        <span class="modal-title">LLM Connection</span>
        <span style="cursor:pointer; color:var(--light); font-size:18px; line-height:1" onclick="closeModal()">&times;</span>
      </div>
      <div class="modal-body">
        <div>
          <div class="input-lbl">LLM API URL</div>
          <input class="field-input" type="text" id="apiUrl" placeholder="http://127.0.0.1:8080/v1/chat/completions">
        </div>
        <div>
          <div class="input-lbl">LLM API Key</div>
          <input class="field-input" type="password" id="apiKey" placeholder="(leave empty for local)">
        </div>
        <div>
          <div class="input-lbl">STT API URL (parakeet.cpp)</div>
          <input class="field-input" type="text" id="sttApiUrl" placeholder="http://localhost:8080">
        </div>
        <div style="font-family:var(--mono); font-size:10px; color:var(--light); margin-top:5px; letter-spacing:0.04em;">
          // Settings are stored in localStorage.
        </div>
      </div>
      <div class="modal-footer">
        <button class="clear-btn" style="width:auto; padding:6px 16px; min-height:0" onclick="closeModal()">Cancel</button>
        <button class="send-btn" style="height:auto; padding:6px 16px" onclick="saveConnection()">Save</button>
      </div>
    </div>
  </div>

  <script type="module">
    const sessionStart = Date.now();
    let audioQueue = [];
    let isPlaying = false;
    let isBusy = false;
    let currentAssistantEl = null;
    let currentReasoningEl = null;
    let sentencesPlayed = 0;
    let charsSynthesized = 0;
    let lastSynthMs = 0;

    const DEFAULT_API_URL = '{{ default_api_url }}';
    const DEFAULT_STT_API_URL = '{{ default_stt_api_url }}';

    document.querySelectorAll('input[type=range]').forEach(el => {
      const min = +el.min, max = +el.max, val = +el.value;
      el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
    });
    window.updateSlider = el => {
      const min = +el.min, max = +el.max, val = +el.value;
      el.style.setProperty('--fill', ((val - min) / (max - min) * 100) + '%');
    };

    setInterval(() => {
      const s = Math.floor((Date.now() - sessionStart) / 1000);
      const m = String(Math.floor(s / 60)).padStart(2, '0');
      document.getElementById('sessionMeta').textContent = 'SES \\u00b7 ' + m + ':' + String(s % 60).padStart(2, '0');
    }, 1000);

    function log(text, type = '') {
      const ls = document.getElementById('logStream');
      const ts = new Date().toISOString().slice(11, 19);
      const el = document.createElement('div');
      el.className = 'log-entry' + (type ? ' ' + type : '');
      el.innerHTML = '<span class="ts">' + ts + '</span>' + text;
      ls.appendChild(el);
      ls.scrollTop = ls.scrollHeight;
      if (ls.children.length > 120) ls.firstChild.remove();
    }
    window.log = log;

    function loadSettings() {
      const url = localStorage.getItem('supertonic_api_url') || DEFAULT_API_URL;
      const key = localStorage.getItem('supertonic_api_key') || '';
      const stt = localStorage.getItem('supertonic_stt_api_url') || DEFAULT_STT_API_URL;
      document.getElementById('apiUrl').value = url;
      document.getElementById('apiKey').value = key;
      document.getElementById('sttApiUrl').value = stt;
    }
    window.saveConnection = function() {
      const url = document.getElementById('apiUrl').value.trim();
      const key = document.getElementById('apiKey').value.trim();
      const stt = document.getElementById('sttApiUrl').value.trim();
      localStorage.setItem('supertonic_api_url', url);
      localStorage.setItem('supertonic_api_key', key);
      localStorage.setItem('supertonic_stt_api_url', stt);
      closeModal();
      log('Connection settings updated.', 'ok');
    };
    window.openModal = function() {
      document.getElementById('settingsModal').classList.remove('hidden');
    };
    window.closeModal = function() {
      document.getElementById('settingsModal').classList.add('hidden');
    };
    document.getElementById('settingsModal').addEventListener('click', e => {
      if (e.target.id === 'settingsModal') closeModal();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });
    loadSettings();

    function syncTopbar() {
      document.getElementById('liveVoice').textContent = document.getElementById('voice').value;
      document.getElementById('liveLang').textContent = document.getElementById('lang').value.toUpperCase();
    }
    window.onVoiceChange = function() {
      syncTopbar();
      log('Voice set to ' + document.getElementById('voice').value);
    };
    window.onLangChange = function() {
      syncTopbar();
      log('Language set to ' + document.getElementById('lang').value.toUpperCase());
    };
    syncTopbar();

    function setStatus(text, state) {
      document.getElementById('statusText').textContent = text;
      const dot = document.getElementById('statusDot');
      dot.className = 'status-dot' + (state === 'active' ? ' active' : state === 'rec' ? ' rec' : '');
    }

    function appendMessage(role, content) {
      const container = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'message ' + role;
      const now = new Date().toTimeString().slice(0, 8);
      let avatar = 'AI', name = 'ASSISTANT';
      if (role === 'user') { avatar = 'USR'; name = 'OPERATOR'; }
      else if (role === 'reasoning') { avatar = 'THK'; name = 'REASONING'; }
      div.innerHTML = '<div class="msg-avatar">' + avatar + '</div><div class="msg-body"><div class="msg-header"><span class="msg-name">' + name + '</span><span class="msg-time">' + now + '</span></div><div class="msg-content"></div></div>';
      const contentDiv = div.querySelector('.msg-content');
      contentDiv.textContent = content || '';
      container.appendChild(div);
      container.scrollTop = container.scrollHeight;
      return div;
    }

    function addOrUpdateAssistant(text) {
      if (!currentAssistantEl) {
        document.getElementById('initOverlay').classList.add('hidden');
        document.getElementById('messages').classList.remove('hidden');
        currentAssistantEl = appendMessage('assistant', '');
      }
      currentAssistantEl.querySelector('.msg-content').textContent += text;
      document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
    }

    function addReasoning(text) {
      if (!currentReasoningEl) {
        currentReasoningEl = appendMessage('reasoning', '');
      }
      currentReasoningEl.querySelector('.msg-content').textContent += text;
      document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
    }

    function attachCursor() {
      if (!currentAssistantEl) return;
      const content = currentAssistantEl.querySelector('.msg-content');
      const old = content.querySelector('.typing-cursor');
      if (old) old.remove();
      const cursor = document.createElement('span');
      cursor.className = 'typing-cursor';
      content.appendChild(cursor);
    }

    function playNext() {
      if (audioQueue.length === 0) { isPlaying = false; return; }
      isPlaying = true;
      const item = audioQueue.shift();
      sentencesPlayed++;
      const audio = new Audio('data:audio/wav;base64,' + item.b64);
      audio.onended = () => playNext();
      audio.onerror = () => { log('Audio playback error', 'warn'); playNext(); };
      audio.play().catch(() => playNext());
    }
    function queueAudio(b64, durMs) {
      audioQueue.push({ b64, durMs });
      if (!isPlaying) playNext();
    }

    window.onInputChange = function() {
      document.getElementById('sendBtn').disabled = !document.getElementById('userInput').value.trim();
    };
    window.autoResize = function(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    };
    window.handleKey = function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendText();
      }
    };

    async function sendToServer(message) {
      isBusy = true;
      document.getElementById('sendBtn').disabled = true;
      document.getElementById('pttBtn').disabled = true;
      setStatus('Streaming', 'active');
      currentAssistantEl = null;
      currentReasoningEl = null;
      let firstText = true;

      let resp;
      try {
        resp = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            message: message,
            lang: document.getElementById('lang').value,
            voice: document.getElementById('voice').value,
            steps: parseInt(document.getElementById('steps').value),
            speed: parseFloat(document.getElementById('speed').value),
            api_url: document.getElementById('apiUrl').value.trim(),
            api_key: document.getElementById('apiKey').value.trim(),
          })
        });
      } catch (err) {
        setStatus('Error', '');
        log('Connection error: ' + err.message, 'warn');
        isBusy = false;
        document.getElementById('pttBtn').disabled = false;
        onInputChange();
        return;
      }

      if (!resp.ok) {
        setStatus('Error ' + resp.status, '');
        log('Server error ' + resp.status, 'warn');
        isBusy = false;
        document.getElementById('pttBtn').disabled = false;
        onInputChange();
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'text') {
                if (firstText) { firstText = false; log('LLM streaming...'); }
                currentReasoningEl = null;
                addOrUpdateAssistant(data.text);
                attachCursor();
              } else if (data.type === 'reasoning') {
                addReasoning(data.text);
              } else if (data.type === 'audio') {
                lastSynthMs = data.synth_ms || 0;
                charsSynthesized += data.chars || 0;
                queueAudio(data.data, data.synth_ms);
                log('Synth \\u00b7 ' + (data.synth_ms || '?') + 'ms \\u00b7 ' + (data.chars || '?') + 'ch', 'ok');
              } else if (data.type === 'error') {
                log('TTS error: ' + data.text, 'warn');
                setStatus('TTS Error', '');
              } else if (data.type === 'done') {
                currentReasoningEl = null;
                if (currentAssistantEl) {
                  const c = currentAssistantEl.querySelector('.typing-cursor');
                  if (c) c.remove();
                }
                setStatus('Ready', '');
                log('Response complete \\u00b7 ' + sentencesPlayed + ' sentence(s)', 'ok');
              }
            } catch(e) { /* ignore */ }
          }
        }
      }

      isBusy = false;
      document.getElementById('pttBtn').disabled = false;
      onInputChange();
    }

    window.sendText = function() {
      if (isBusy) return;
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      document.getElementById('initOverlay').classList.add('hidden');
      document.getElementById('messages').classList.remove('hidden');
      appendMessage('user', text);
      input.value = '';
      autoResize(input);
      onInputChange();
      sendToServer(text);
    };

    window.clearChat = async function() {
      document.getElementById('messages').innerHTML = '';
      document.getElementById('messages').classList.add('hidden');
      document.getElementById('initOverlay').classList.remove('hidden');
      sentencesPlayed = 0;
      charsSynthesized = 0;
      lastSynthMs = 0;
      currentAssistantEl = null;
      currentReasoningEl = null;
      try { await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:'/clear'})}); } catch(e){}
      log('Conversation cleared.', 'ok');
    };

    // STT via parakeet.cpp (local, on-device)
    let pttHeld = false;
    let spaceHeld = false;
    let sttPrefix = '';
    let mediaStream = null;
    let audioContext = null;
    let scriptProcessor = null;
    let sttChunks = [];

    function encodeWAV(samples, sampleRate) {
      const buf = new ArrayBuffer(44 + samples.length * 2);
      const v = new DataView(buf);
      const w = (s, o) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
      w('RIFF', 0); v.setUint32(4, 36 + samples.length * 2, true); w('WAVE', 8);
      w('fmt ', 12); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
      v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
      w('data', 36); v.setUint32(40, samples.length * 2, true);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return new Blob([buf], { type: 'audio/wav' });
    }

    async function startPTT() {
      if (isBusy) return;
      const current = document.getElementById('userInput').value;
      sttPrefix = current ? (current.endsWith(' ') ? current : current + ' ') : '';
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });
        audioContext = new AudioContext({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        sttChunks = [];
        scriptProcessor.onaudioprocess = (e) => {
          sttChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        pttHeld = true;
        document.getElementById('pttBtn').classList.add('recording');
        setStatus('Recording', 'rec');
        log('Recording via parakeet.cpp STT...', 'hl');
      } catch (e) {
        log('Mic error: ' + e.message, 'warn');
      }
    }

    async function stopPTT() {
      pttHeld = false;
      if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
      if (audioContext) { await audioContext.close(); audioContext = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      document.getElementById('pttBtn').classList.remove('recording');
      if (sttChunks.length === 0) {
        setStatus('Ready', '');
        return;
      }
      setStatus('Transcribing', 'active');
      const total = sttChunks.reduce((s, c) => s + c.length, 0);
      const samples = new Float32Array(total);
      let offset = 0;
      for (const c of sttChunks) { samples.set(c, offset); offset += c.length; }
      const wav = encodeWAV(samples, 16000);
      sttChunks = [];
      try {
        const formData = new FormData();
        formData.append('file', wav, 'recording.wav');
        formData.append('lang', document.getElementById('lang').value);
        formData.append('stt_api', document.getElementById('sttApiUrl').value.trim());
        const resp = await fetch('/api/stt', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.text) {
          const input = document.getElementById('userInput');
          input.value = (sttPrefix + data.text).replace(/^\\s+|\\s+$/g, '');
          autoResize(input);
          onInputChange();
          log('STT: ' + data.text.substring(0, 50) + (data.text.length > 50 ? '...' : ''), 'ok');
        } else if (data.error) {
          log('STT error: ' + data.error, 'warn');
        }
      } catch (e) {
        log('STT error: ' + e.message, 'warn');
      }
      sttPrefix = '';
      setStatus('Ready', '');
    }

    const pttBtn = document.getElementById('pttBtn');
    pttBtn.addEventListener('mousedown', e => { e.preventDefault(); startPTT(); });
    pttBtn.addEventListener('mouseup', e => { e.preventDefault(); stopPTT(); });
    pttBtn.addEventListener('mouseleave', () => { if (pttHeld) stopPTT(); });
    pttBtn.addEventListener('touchstart', e => { e.preventDefault(); startPTT(); }, {passive: false});
    pttBtn.addEventListener('touchend', e => { e.preventDefault(); stopPTT(); }, {passive: false});
    pttBtn.addEventListener('touchcancel', () => stopPTT());

    document.addEventListener('keydown', e => {
      if (e.code === 'Space' && !spaceHeld && document.activeElement !== document.getElementById('userInput')) {
        spaceHeld = true;
        e.preventDefault();
        startPTT();
      }
    });
    document.addEventListener('keyup', e => {
      if (e.code === 'Space' && spaceHeld) {
        spaceHeld = false;
        e.preventDefault();
        stopPTT();
      }
    });

    setStatus('Ready', '');
    log('Supertonic voice chat ready \u2014 STT via parakeet.cpp.', 'ok');
    document.getElementById('userInput').focus();
  </script>
</body>
</html>"""



@app.route("/")
def index():
    return render_template_string(
        HTML,
        default_api_url=config["api_url"],
        default_stt_api_url=config["stt_api_url"],
    )


def main():
    global tts, style, config

    parser = argparse.ArgumentParser(description="Supertonic Voice Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7777, help="Port")
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

    print("🚀 Loading Supertonic TTS...")
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name=config["voice"])

    print(f"""
╔══════════════════════════════════════════╗
║   🎤 Supertonic Voice Chat              ║
║   Open: http://{args.host}:{args.port}          ║
║   LLM:  {args.api}        ║
║   STT:  {args.stt_api}           ║
║   Voice: {args.voice}  |  Lang: {args.lang}      ║
║   Steps: {args.steps}  |  Speed: {args.speed}          ║
╚══════════════════════════════════════════╝
""")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
