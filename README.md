# Supertonic — Voice Chat

Local voice assistant: mic → STT → LLM → TTS → speaker. Everything on-device.

```
Browser mic ──→ parakeet.cpp ──→ llama.cpp ──→ Supertonic TTS ──→ audio
                (STT)            (LLM)         (ONNX Runtime)
                :8081            :8080         (in-process)
```

## Quick Start

```bash
# 1. Download STT model (one-time)
mkdir -p models
curl -L -o models/tdt_ctc-110m-q5_k.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q5_k.gguf

# 2. Start everything (llama + parakeet + chat UI)
docker compose up -d

# 3. Open http://localhost:7777
```

The first start pulls the LLM weights (~2.6 GB) into the `hf_cache` volume.
Subsequent starts are instant.

## Stack

| Role | Tech | Port |
|------|------|------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (NVIDIA NeMo → ggml) | `:8081` |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) — `unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL` (auto-downloaded) | `:8080` |
| TTS | [Supertonic 3](https://github.com/supertone-inc/supertonic) (ONNX Runtime) | — |
| UI | Flask + vanilla JS | `:7777` |

### Tweak the LLM

The LLM is configured in `docker-compose.yml` under the `llama` service. To swap
models, just change the `-hf` flag, e.g.:

```yaml
command: >
  llama serve
    --hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL
    --host 0.0.0.0
    --port 8080
```

For GPU acceleration, uncomment the `deploy.resources` block under the `llama`
service and run with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
installed. The `llama` binary inside the image was built CPU-only (Docker build
skips CUDA probing), so you'd want to drop the `SKIP_CUDA=1` env in the
`Dockerfile` to let the installer pick a CUDA build on a GPU host.

For gated HuggingFace repos, set `HF_TOKEN` in the `llama` service environment
and the installer's `curl` will pass it through automatically.

## Usage

Hold mic or Space to talk, release to transcribe, Enter to send. Edit voice, language, and system prompt in the left panel.
