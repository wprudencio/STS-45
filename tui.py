#!/usr/bin/env python3
"""
Supertonic — Voice Chat (TUI)

Beautiful terminal interface matching the web UI design language:
light theme, orange accent, Geist-inspired mono layout, live metrics.
"""

import argparse
import sys
import time
import threading
import re
from datetime import datetime

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich import box

from chat import StreamingPlayer, SentenceBuffer, LLMClient, LLAMA_API, SYS_PROMPT
from supertonic import TTS


# ── Theme (matches web UI) ─────────────────────────────────────
ORANGE = "#ea6626"
SOFT = "#e8ab8f"
DEEP = "#020202"
CHARCOAL = "#303030"
MID = "#545352"
LIGHT = "#737270"
BG = "#eeeeee"
BG2 = "#f6f6f6"
OK = "#7a9e7e"
ERR = "#d04848"

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Supertonic voice chat (TUI)")
    p.add_argument("--lang", default="en")
    p.add_argument("--voice", default="M1")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.15)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--api", default=LLAMA_API)
    p.add_argument("--model", default="default")
    p.add_argument("--no-llm", action="store_true",
                   help="Echo mode (no LLM, just TTS the input)")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def banner():
    console.print()
    title = Text()
    title.append("SUPERTONIC", style="bold " + ORANGE)
    title.append("  //  ", style=LIGHT)
    title.append("VOICE CHAT", style="bold " + ORANGE)
    console.print(Panel(
        Align.center(title),
        box=box.DOUBLE, border_style=ORANGE, padding=(0, 2),
    ))
    console.print()
    console.print("  Local TTS · streaming LLM · sentence-by-sentence synthesis",
                  style=SOFT)
    console.print()


def status_bar(state, voice, lang, synth_ms, queue_n, played, chars):
    """One-line status bar like the web UI topbar."""
    dot_color = ORANGE if state in ("Streaming", "Synthesizing") else (
        SOFT if state == "Listening" else LIGHT)
    t = Text()
    t.append("●", style=dot_color)
    t.append(" ")
    if state == "Ready":
        t.append(state.upper(), style=LIGHT)
    else:
        t.append(state.upper(), style="bold " + ORANGE)
    t.append("    ")
    t.append("VOICE", style=LIGHT)
    t.append(" " + str(voice), style=CHARCOAL)
    t.append("    ")
    t.append("LANG", style=LIGHT)
    t.append(" " + str(lang).upper(), style=CHARCOAL)
    t.append("    ")
    t.append("SYNTH", style=LIGHT)
    t.append(" " + str(synth_ms) + "ms", style=CHARCOAL)
    t.append("    ")
    t.append("QUEUE", style=LIGHT)
    t.append(" " + str(queue_n), style=CHARCOAL)
    t.append("    ")
    t.append("PLAYED", style=LIGHT)
    t.append(" " + str(played), style=CHARCOAL)
    t.append("    ")
    t.append("CHARS", style=LIGHT)
    t.append(" " + str(chars), style=CHARCOAL)
    return t


def print_status(**kwargs):
    console.print(status_bar(**kwargs))
    console.print()


def role_header(role):
    if role == "user":
        console.print("  ● USER", style="bold " + ORANGE)
    elif role == "assistant":
        console.print("  ● ASSISTANT", style="bold " + ORANGE)
    elif role == "reasoning":
        console.print("  ● REASONING", style=SOFT)
    console.print()


def print_help():
    rows = [
        ("Commands", True, None, ORANGE),
        ("  :clear    ", True, "clear chat history", None),
        ("  :lang X   ", True, "set language (en, pt, es...)", None),
        ("  :voice X  ", True, "set voice (M1-M5, F1-F5)", None),
        ("  :steps N  ", True, "set diffusion steps (2-12)", None),
        ("  :speed F  ", True, "set playback speed (0.7-2.0)", None),
        ("  :stats    ", True, "show current settings", None),
        ("  :help     ", True, "show this help", None),
        ("  :quit     ", True, "exit", None),
    ]
    body = Text()
    for i, (text, is_key, _desc, color) in enumerate(rows):
        if i == 0:
            body.append(text + "\n", style="bold " + ORANGE)
        else:
            body.append(text, style="bold " + CHARCOAL)
            body.append(_desc + "\n", style=LIGHT)
    console.print(Panel(
        body,
        border_style=LIGHT, box=box.ROUNDED, padding=(0, 2),
    ))


def stream_text(token_iter, on_sentence):
    """Stream tokens to console, calling on_sentence(s) when a sentence is detected."""
    buf = SentenceBuffer()
    full = []
    reasoning_started = False
    for kind, token in token_iter:
        if kind == "reasoning":
            if not reasoning_started:
                console.print()
                role_header("reasoning")
                reasoning_started = True
            console.print(token, end="", style=SOFT)
            continue
        if reasoning_started:
            console.print()
            role_header("assistant")
            reasoning_started = False
        console.print(token, end="", style=CHARCOAL)
        full.append(token)
        s = buf.add(token)
        if s and len(s) > 3:
            on_sentence(s)
    s = buf.flush()
    if s and len(s) > 3:
        on_sentence(s)
    return "".join(full), reasoning_started


def main():
    args = parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        sys.exit(0)

    banner()

    # ── Load TTS ──
    with console.status("Loading Supertonic TTS…", spinner="dots"):
        tts = TTS(auto_download=True)
        style = tts.get_voice_style(voice_name=args.voice)
    player = StreamingPlayer(tts.sample_rate, device=args.device)

    # ── LLM ──
    llm = None
    if not args.no_llm:
        with console.status("Connecting to " + args.api + "…", spinner="dots"):
            try:
                llm = LLMClient(args.api, args.model)
            except Exception as e:
                console.print("  LLM error: " + str(e), style=ERR)
                console.print("  Falling back to echo mode (use :quit to exit)", style=LIGHT)
                console.print()
                llm = None

    lang = args.lang
    voice = args.voice
    steps = args.steps
    speed = args.speed

    state = "Ready"
    synth_ms = 0
    queue_n = 0
    played = 0
    chars = 0

    print_status(state=state, voice=voice, lang=lang, synth_ms="—",
                 queue_n=0, played=0, chars=0)
    console.print("  Type a message and press Enter. Type :help for commands.", style=LIGHT)
    console.print()

    while True:
        try:
            console.print("  > ", style="bold " + ORANGE, end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print("  ● Goodbye.", style=ORANGE)
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input in (":quit", ":q", ":exit"):
            console.print("  ● Goodbye.", style=ORANGE)
            break

        if user_input == ":help":
            console.print()
            print_help()
            console.print()
            continue

        if user_input == ":clear" and llm:
            llm.clear_history()
            console.print("  ✓ History cleared.", style=OK)
            console.print()
            continue

        if user_input == ":stats":
            mode = "LLM streaming" if llm else "Echo (TTS only)"
            body = Text()
            body.append("Voice  ", style=LIGHT)
            body.append(str(voice) + "\n", style=CHARCOAL)
            body.append("Lang   ", style=LIGHT)
            body.append(str(lang) + "\n", style=CHARCOAL)
            body.append("Steps  ", style=LIGHT)
            body.append(str(steps) + "\n", style=CHARCOAL)
            body.append("Speed  ", style=LIGHT)
            body.append(str(speed) + "\n", style=CHARCOAL)
            body.append("API    ", style=LIGHT)
            body.append(str(args.api) + "\n", style=CHARCOAL)
            body.append("Mode   ", style=LIGHT)
            body.append(mode, style=CHARCOAL)
            console.print(Panel(
                body, border_style=LIGHT, box=box.ROUNDED,
                title="Status", title_align="left", padding=(0, 2),
            ))
            console.print()
            continue

        if user_input.startswith(":"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0] if parts else ""
            val = parts[1] if len(parts) > 1 else ""
            try:
                if cmd == "lang" and val:
                    lang = val
                    console.print("  ✓ Lang → " + lang, style=OK)
                    console.print()
                elif cmd == "voice" and val:
                    voice = val
                    style = tts.get_voice_style(voice_name=voice)
                    console.print("  ✓ Voice → " + voice, style=OK)
                    console.print()
                elif cmd == "steps" and val:
                    steps = int(val)
                    console.print("  ✓ Steps → " + str(steps), style=OK)
                    console.print()
                elif cmd == "speed" and val:
                    speed = float(val)
                    console.print("  ✓ Speed → " + str(speed), style=OK)
                    console.print()
                else:
                    console.print("  ✗ Unknown: :" + cmd, style=ERR)
                    console.print()
            except ValueError:
                console.print("  ✗ Invalid value", style=ERR)
                console.print()
            continue

        # ── User message ──
        role_header("user")
        console.print("  " + user_input)
        console.print()

        # ── Generate ──
        player.reset()
        played = 0
        chars = 0
        synth_ms = 0
        sentence_count = [0]
        total_synth_ms = [0.0]

        if llm:
            state = "Streaming"
            role_header("assistant")

            def synth_chunk(text, seq_id):
                nonlocal synth_ms
                t0 = time.time()
                try:
                    wav, dur = tts.synthesize(
                        text=text,
                        lang=lang,
                        voice_style=style,
                        total_steps=steps,
                        speed=speed,
                    )
                    dt_ms = int((time.time() - t0) * 1000)
                    synth_ms = dt_ms
                    total_synth_ms[0] += dt_ms
                    chars_now = len(text)
                    console.print(" · synth " + str(dt_ms) + "ms · " + str(chars_now) + "ch",
                                  end="", style=SOFT)
                    player.enqueue(seq_id, wav)
                except Exception as e:
                    console.print(" TTS error: " + str(e), end="", style=ERR)

            seq = [0]

            def on_sentence(s):
                sentence_count[0] += 1
                sid = seq[0]
                seq[0] += 1
                threading.Thread(target=synth_chunk, args=(s, sid), daemon=True).start()

            t0 = time.time()
            try:
                full, had_reasoning = stream_text(llm.stream(user_input), on_sentence)
            except requests.ConnectionError:
                console.print()
                console.print("  ✗ Cannot reach LLM at " + args.api, style=ERR)
                console.print()
                state = "Ready"
                continue
            except Exception as e:
                console.print()
                console.print("  ✗ Error: " + str(e), style=ERR)
                console.print()
                state = "Ready"
                continue
            llm_latency = (time.time() - t0)

            if had_reasoning:
                console.print()
            console.print()
            player.wait_done()
            played = sentence_count[0]
            chars = len(full)
            state = "Ready"
            avg_synth = (total_synth_ms[0] / played) if played else 0
            print_status(state=state, voice=voice, lang=lang,
                         synth_ms=int(avg_synth) if played else "—",
                         queue_n=player._q.qsize() if hasattr(player, "_q") else 0,
                         played=played, chars=chars)
            console.print("  LLM " + ("%.1f" % llm_latency) + "s · " + str(len(full)) +
                          " chars · " + str(played) + " sentence(s)",
                          style=LIGHT)
            console.print()

        else:
            # Echo mode
            state = "Synthesizing"
            role_header("assistant")
            t0 = time.time()
            try:
                wav, dur = tts.synthesize(
                    text=user_input, lang=lang, voice_style=style,
                    total_steps=steps, speed=speed,
                )
                dt_ms = int((time.time() - t0) * 1000)
                synth_ms = dt_ms
                chars = len(user_input)
                console.print("  " + user_input + "  · synth " + str(dt_ms) + "ms",
                              style=CHARCOAL)
                player.enqueue(0, wav)
                player.wait_done()
                played = 1
                state = "Ready"
                console.print()
                print_status(state=state, voice=voice, lang=lang,
                             synth_ms=synth_ms, queue_n=0, played=played, chars=chars)
            except Exception as e:
                console.print("  TTS error: " + str(e), style=ERR)
                state = "Ready"
            console.print()


if __name__ == "__main__":
    main()
