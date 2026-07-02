#!/usr/bin/env python3
"""
Supertonic Realtime Voice
WebSocket-based realtime voice conversation with local pipeline:
mic → VAD → STT (parakeet.cpp) → LLM (llama.cpp) → TTS (Supertonic) → speaker

Usage:
  python3 realtime_server.py --host 0.0.0.0 --port 7777
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import wave
from typing import Optional

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from supertonic import TTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("realtime")

# ── Config ──────────────────────────────────────────────────────────────────
LLAMA_API = os.environ.get("LLAMA_API", "http://127.0.0.1:8080/v1/chat/completions")
STT_API = os.environ.get("STT_API", "http://localhost:8080")
INPUT_SAMPLE_RATE = 16000

SYS_PROMPT = (
    "You are a friendly, helpful voice assistant. "
    "Keep answers concise, warm, and natural for speech. "
    "Use short sentences. Avoid markdown, lists, URLs, or emojis. "
    "Respond in the same language the user speaks to you."
)

LANG_MAP = {
    "en": "en", "pt": "pt", "es": "es", "fr": "fr",
    "de": "de", "ja": "ja", "ko": "ko",
}

HERE = os.path.dirname(os.path.abspath(__file__))

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Supertonic Realtime Voice")

tts: Optional[TTS] = None
tts_lock = threading.Lock()
tts_ready = threading.Event()


def load_tts():
    global tts
    logger.info("Loading Supertonic TTS model (this may download ~260MB on first run)...")
    try:
        tts = TTS()
        voices = list(tts.voice_style_names)
        logger.info("TTS loaded — %d voices: %s", len(voices), voices)
        tts_ready.set()
    except Exception as e:
        logger.error("Failed to load TTS: %s", e)


# ── Audio helpers ───────────────────────────────────────────────────────────

def wav_to_base64(wav: np.ndarray, sample_rate: int) -> str:
    mono = wav.squeeze()
    i16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(i16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    mono = wav.squeeze()
    i16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(i16.tobytes())
    return buf.getvalue()


def base64_to_numpy(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0


# ── VAD ─────────────────────────────────────────────────────────────────────

class VAD:
    """RMS-based Voice Activity Detector with state machine."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.threshold = 0.012
        self.silence_ms = 800
        self.speech_confirm_ms = 120
        self.min_speech_ms = 400

        self._buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._frame_size = 0

    def _compute_limits(self, frame_size: int) -> None:
        if frame_size == self._frame_size:
            return
        self._frame_size = frame_size
        fps = self.sample_rate / frame_size
        self._silence_limit = max(1, int(self.silence_ms / 1000 * fps))
        self._speech_confirm = max(1, int(self.speech_confirm_ms / 1000 * fps))
        self._min_speech = max(5, int(self.min_speech_ms / 1000 * fps))

    def process(self, audio: np.ndarray) -> tuple:
        """Returns (speech_ended: bool, speech_audio: np.ndarray | None)."""
        self._compute_limits(len(audio))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        if rms >= self.threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            if self._is_speaking:
                self._buffer.append(audio)
            elif self._speech_frames >= self._speech_confirm:
                self._is_speaking = True
                self._buffer.append(audio)
        else:
            self._silence_frames += 1
            if self._is_speaking:
                self._buffer.append(audio)
                if self._silence_frames >= self._silence_limit:
                    speech = np.concatenate(self._buffer) if self._buffer else None
                    self._reset()
                    min_samples = int(self.min_speech_ms / 1000 * self.sample_rate)
                    if speech is not None and len(speech) >= min_samples:
                        return True, speech
            else:
                self._buffer.append(audio)
                max_buf = self._speech_confirm * 4
                while len(self._buffer) > max_buf:
                    self._buffer.pop(0)
                self._speech_frames = max(0, self._speech_frames - 1)

        return False, None

    def _reset(self):
        self._is_speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buffer.clear()

    @property
    def speaking(self) -> bool:
        return self._is_speaking


# ── STT ─────────────────────────────────────────────────────────────────────

async def transcribe(audio: np.ndarray, lang: str) -> str:
    wav_bytes = wav_to_bytes(audio, INPUT_SAMPLE_RATE)
    parakeet_lang = LANG_MAP.get(lang, "en")
    url = f"{STT_API.rstrip('/')}/v1/audio/transcriptions"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language": parakeet_lang, "response_format": "json"},
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


# ── LLM ─────────────────────────────────────────────────────────────────────

async def stream_llm(messages: list) -> str:
    payload = {
        "model": "default",
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    full: list[str] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", LLAMA_API, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                d = line[6:]
                if d.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                    c = (chunk.get("choices", [{}])[0]
                          .get("delta", {})
                          .get("content", ""))
                    if c:
                        full.append(c)
                except json.JSONDecodeError:
                    continue
    return "".join(full)


# ── TTS ─────────────────────────────────────────────────────────────────────

def synthesize(text: str, voice: str, lang: str, steps: int, speed: float):
    tts_ready.wait()
    with tts_lock:
        vs = tts.get_voice_style(voice_name=voice)
        wav, _dur = tts.synthesize(
            text=text,
            lang=lang,
            voice_style=vs,
            total_steps=steps,
            speed=speed,
        )
    return wav, tts.sample_rate


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── WebSocket handler ───────────────────────────────────────────────────────

@app.websocket("/v1/realtime")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    vad = VAD(INPUT_SAMPLE_RATE)
    voice = "M1"
    lang = "en"
    steps = 5
    speed = 1.15
    history: list[dict] = [{"role": "system", "content": SYS_PROMPT}]

    async def emit(data: dict):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    async def set_status(state: str):
        await emit({"type": "status", "state": state})

    await set_status("idle")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type", "")

            if t == "ping":
                await emit({"type": "pong"})

            elif t == "audio":
                data = msg.get("data", "")
                if not data:
                    continue
                audio = base64_to_numpy(data)

                speech_ended, speech_audio = vad.process(audio)

                if vad.speaking and not speech_ended:
                    await set_status("listening")
                    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                    await emit({"type": "audio.level", "rms": rms})

                if speech_ended and speech_audio is not None:
                    await set_status("processing")

                    try:
                        text = await transcribe(speech_audio, lang)
                        if not text:
                            logger.info("STT returned empty")
                            await set_status("idle")
                            continue

                        logger.info("Transcribed: %r", text)
                        await emit({"type": "transcript", "role": "user", "text": text})

                        history.append({"role": "user", "content": text})
                        response = await stream_llm(history)
                        if not response:
                            logger.info("LLM returned empty")
                            await set_status("idle")
                            continue

                        history.append({"role": "assistant", "content": response})
                        logger.info("LLM: %.80s...", response)

                        await emit({"type": "transcript", "role": "assistant", "text": response})

                        await set_status("speaking")
                        sentences = split_sentences(response)
                        if not sentences:
                            sentences = [response]

                        loop = asyncio.get_event_loop()
                        for sentence in sentences:
                            try:
                                wav, sr = await loop.run_in_executor(
                                    None, synthesize, sentence, voice, lang, steps, speed
                                )
                                b64 = wav_to_base64(wav, sr)
                                await emit({"type": "audio.chunk", "data": b64})
                            except Exception as e:
                                logger.error("TTS error for sentence %r: %s", sentence[:40], e)
                                continue

                        await emit({"type": "audio.done"})
                        await set_status("idle")

                    except httpx.HTTPError as e:
                        logger.error("HTTP error in pipeline: %s", e)
                        await emit({"type": "error", "message": f"Service unavailable: {e}"})
                        await set_status("error")
                    except Exception as e:
                        logger.error("Pipeline error: %s", e)
                        await emit({"type": "error", "message": str(e)})
                        await set_status("error")

            elif t == "session.update":
                if "voice" in msg:
                    voice = msg["voice"]
                if "lang" in msg:
                    lang = msg.get("lang", lang)
                if "instructions" in msg and msg["instructions"]:
                    history[0] = {"role": "system", "content": msg["instructions"]}
                if "steps" in msg:
                    steps = int(msg.get("steps", steps))
                if "speed" in msg:
                    speed = float(msg.get("speed", speed))
                logger.info("Session: voice=%s lang=%s steps=%d speed=%.2f",
                           voice, lang, steps, speed)

            elif t == "clear":
                history = [{"role": "system", "content": SYS_PROMPT}]
                await emit({"type": "cleared"})

            elif t == "stop":
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)


# ── HTTP routes ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    path = os.path.join(HERE, "templates", "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    with open(path) as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "tts_ready": tts is not None}


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    global STT_API, LLAMA_API

    parser = argparse.ArgumentParser(description="Supertonic Realtime Voice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--stt-api", default=STT_API)
    parser.add_argument("--api", default=LLAMA_API)
    args = parser.parse_args()

    STT_API = args.stt_api
    LLAMA_API = args.api

    t = threading.Thread(target=load_tts, daemon=True)
    t.start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
