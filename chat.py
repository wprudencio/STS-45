#!/usr/bin/env python3
"""
Supertonic Chat + LLM — conversa com IA local via llama.cpp e ouve a resposta.
Streaming real: cada sentença é sintetizada e tocada assim que fica pronta.
"""

import argparse
import json
import sys
import time
import re
import threading
import queue
from typing import Optional

import numpy as np
import sounddevice as sd
import requests
from supertonic import TTS


LLAMA_API = "http://127.0.0.1:8080/v1/chat/completions"
SYS_PROMPT = (
    "You are a friendly, helpful assistant. Respond in the same language as the user. "
    "Keep answers concise and natural for text-to-speech. "
    "Avoid markdown, lists, URLs, or special formatting. "
    "Use short to medium sentences. Avoid asterisks and emojis."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Chat com IA local + TTS em tempo real")
    parser.add_argument("--lang", default="en", help="Idioma padrão (en, pt, es, etc)")
    parser.add_argument("--voice", default="M1", help="Voz: M1-M5, F1-F5")
    parser.add_argument("--steps", type=int, default=5, help="Passos de denoising (5=rápido)")
    parser.add_argument("--speed", type=float, default=1.15, help="Velocidade da fala")
    parser.add_argument("--device", type=int, default=None, help="ID do dispositivo de áudio")
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos de áudio")
    parser.add_argument("--model", default="default", help="Nome do modelo no llama.cpp")
    parser.add_argument("--api", default=LLAMA_API, help="URL da API llama.cpp")
    parser.add_argument("--no-llm", action="store_true", help="Modo chat simples (sem LLM)")
    return parser.parse_args()


class StreamingPlayer:
    """
    Toca áudio em ordem, assim que cada chunk fica pronto, sem bloquear.
    Chunks podem chegar fora de ordem; o player reordena e toca sequencial.
    """

    def __init__(self, sample_rate: int, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._playing = False
        # Thread única que toca em sequência
        self._thread = threading.Thread(target=self._player_loop, daemon=True)
        self._thread.start()
        self.reset()

    def reset(self):
        """Reseta estado entre interações."""
        with self._lock:
            self._next_seq = 0
            self._buffer: dict[int, np.ndarray] = {}

    def enqueue(self, seq: int, wav: np.ndarray):
        """Recebe um chunk de áudio (pode chegar fora de ordem)."""
        with self._lock:
            self._buffer[seq] = wav
            self._play_pending()

    def _play_pending(self):
        """Empurra chunks prontos em sequência para a fila de playback."""
        while self._next_seq in self._buffer:
            w = self._buffer.pop(self._next_seq)
            self._next_seq += 1
            self._q.put(w)
            self._playing = True

    def _player_loop(self):
        """Thread única que toca os chunks, um por um."""
        while True:
            wav = self._q.get()
            if wav is None:
                break
            try:
                sd.play(wav.squeeze(), samplerate=self.sample_rate, device=self.device)
                sd.wait()
            except Exception as e:
                print(f"\n  [playback error: {e}]")
            finally:
                with self._lock:
                    self._playing = self._q.qsize() > 0
                    self._cond.notify_all()

    def wait_done(self):
        """Espera todos os chunks terminarem de tocar."""
        with self._cond:
            self._play_pending()
            while self._playing or self._buffer:
                self._cond.wait(timeout=0.5)


class SentenceBuffer:
    """Acumula tokens e detecta quando uma sentença está completa."""

    def __init__(self):
        self.buffer = ""
        self._sentence_end = re.compile(r"[.!?…]\s*$")

    def add(self, token: str) -> Optional[str]:
        self.buffer += token
        if len(self.buffer) > 180:
            s = self.buffer.strip()
            self.buffer = ""
            return s
        if self._sentence_end.search(self.buffer):
            # Não divide em abreviações comuns
            if not re.search(
                r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|e\.g|i\.e|vs|Inc|Ltd|St|Ave|Blvd)\.$",
                self.buffer,
            ):
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


class LLMClient:
    """Cliente streaming para llama.cpp (OpenAI-compatível)."""

    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model
        self.messages = [{"role": "system", "content": SYS_PROMPT}]

    def clear_history(self):
        self.messages = [{"role": "system", "content": SYS_PROMPT}]

    def stream(self, user_msg: str, max_tokens: int = 512):
        """
        Gera tokens um por um via SSE.
        Yield (kind, text) onde kind é 'content' ou 'reasoning'.
        Só retorna 'content' para síntese; 'reasoning' é mostrado mas não falado.
        """
        self.messages.append({"role": "user", "content": user_msg})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        content_parts = []
        has_content = False

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                content = delta.get("content")

                # Se o modelo bota a resposta em reasoning_content (reasoning models)
                if reasoning and not has_content:
                    yield ("reasoning", reasoning)
                elif content:
                    has_content = True
                    content_parts.append(content)
                    yield ("content", content)

        full = "".join(content_parts)
        self.messages.append({"role": "assistant", "content": full})


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # --- Carrega TTS --- #
    print("🚀 Loading Supertonic TTS...")
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name=args.voice)
    player = StreamingPlayer(tts.sample_rate, device=args.device)

    # --- LLM --- #
    llm = None
    if not args.no_llm:
        print(f"🤖 Connecting to {args.api}")
        llm = LLMClient(args.api, args.model)
        print(f"   Model: {llm.model}")
    else:
        print("💬 Simple chat mode (no LLM)")

    print(f"🎤 Voice: {args.voice} | Lang: {args.lang} | Steps: {args.steps} | Speed: {args.speed}")
    print()
    print("Type something and press Enter. Special commands:")
    print("  :clear  — clear chat history")
    print("  :lang   — change language (en, pt, es, etc)")
    print("  :voice  — change voice (M1-M5, F1-F5)")
    print("  :steps  — change quality/speed (5-12)")
    print("  :speed  — change playback speed")
    print("  :stats  — show current settings")
    print("  :quit   — exit")
    print()

    lang = args.lang
    voice = args.voice
    steps = args.steps
    speed = args.speed

    while True:
        try:
            user_input = input("🤔 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # --- Comandos --- #
        if user_input in (":quit", ":q"):
            print("👋 Goodbye!")
            break

        if user_input == ":clear" and llm:
            llm.clear_history()
            print("🧹 History cleared!")
            continue

        if user_input == ":stats":
            print(f"  Lang: {lang} | Voice: {voice} | Steps: {steps} | Speed: {speed}")
            continue

        if user_input.startswith(":"):
            cmd, *rest = user_input[1:].split()
            val = " ".join(rest) if rest else ""
            if cmd == "lang" and val:
                lang = val
                print(f"🌍 Language: {lang}")
            elif cmd == "voice" and val:
                try:
                    voice = val
                    style = tts.get_voice_style(voice_name=voice)
                    print(f"🎤 Voice: {voice}")
                except Exception as e:
                    print(f"❌ {e}")
            elif cmd == "steps" and val:
                try:
                    steps = int(val)
                    print(f"⚙️  Steps: {steps}")
                except ValueError:
                    print("⚠️  Invalid number")
            elif cmd == "speed" and val:
                try:
                    speed = float(val)
                    print(f"⚙️  Speed: {speed}")
                except ValueError:
                    print("⚠️  Invalid number")
            else:
                print(f"⚠️  Unknown command: {user_input}")
            continue

        # --- LLM + TTS streaming --- #
        try:
            # Reseta o player para nova interação
            player.reset()
            
            if llm:
                print("🤖 ", end="", flush=True)
                buf = SentenceBuffer()
                seq = [0]
                synth_in_flight = []
                reasoning_shown = False

                def synthesize_chunk(text_chunk, seq_id):
                    try:
                        wav, dur = tts.synthesize(
                            text=text_chunk,
                            lang=lang,
                            voice_style=style,
                            total_steps=steps,
                            speed=speed,
                        )
                        player.enqueue(seq_id, wav)
                    except Exception as e:
                        print(f"\n  [TTS error: {e}]")

                # Stream tokens
                for kind, token in llm.stream(user_input):
                    if kind == "reasoning":
                        if not reasoning_shown:
                            print("\n🧠 thinking:", end=" ", flush=True)
                            reasoning_shown = True
                        print(token, end="", flush=True)
                        continue

                    if reasoning_shown:
                        print()
                        reasoning_shown = False

                    # Content token → mostra e acumula
                    print(token, end="", flush=True)
                    sentence = buf.add(token)
                    if sentence is not None and len(sentence) > 3:
                        sid = seq[0]
                        seq[0] += 1
                        t = threading.Thread(
                            target=synthesize_chunk,
                            args=(sentence, sid),
                            daemon=True,
                        )
                        t.start()
                        synth_in_flight.append(t)

                print()  # nova linha após a resposta

                # Último pedaço
                sentence = buf.flush()
                if sentence and len(sentence) > 3:
                    sid = seq[0]
                    seq[0] += 1
                    t = threading.Thread(
                        target=synthesize_chunk,
                        args=(sentence, sid),
                        daemon=True,
                    )
                    t.start()
                    synth_in_flight.append(t)

                # Espera todas as TTS terminarem
                for t in synth_in_flight:
                    t.join()

                # Espera player terminar de tocar
                player.wait_done()

            else:
                # Simple mode: just TTS
                t_start = time.time()
                wav, duration = tts.synthesize(
                    text=user_input,
                    lang=lang,
                    voice_style=style,
                    total_steps=steps,
                    speed=speed,
                )
                t_gen = time.time() - t_start
                print(f"✅ {duration[0]:.1f}s audio in {t_gen:.2f}s ({(t_gen/max(duration[0],0.01)):.2f}x RTF)")
                player.enqueue(0, wav)
                player.wait_done()

        except requests.ConnectionError:
            print("❌ Cannot connect to llama.cpp. Is it running?")
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
