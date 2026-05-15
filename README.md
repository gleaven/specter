# SPECTER — Edge Inference Optimisation Workbench

> Run head-to-head LLM benchmarks on your local GPU — tokens/sec, TTFT,
> GPU memory, power draw, and accuracy — across Ollama, vLLM, and any
> OpenAI-compatible endpoint, with a persistent leaderboard.

---

## What this demo is

SPECTER is a benchmarking workbench for running **the same prompt
through multiple LLMs back-to-back** and comparing what comes out the
other end on five axes:

1. **Throughput** — sustained tokens/second during generation.
2. **Latency** — time-to-first-token (TTFT) measured from the streaming
   response.
3. **GPU footprint** — VRAM used while the model is resident, polled
   from `nvidia-smi` (and supplemented from Ollama's `/api/ps` on
   unified-memory GPUs where `nvidia-smi` reports `[N/A]` for memory).
4. **Power draw** — wall-clock watts pulled while generating.
5. **Quality** — substring-match accuracy + structured-output scoring
   against curated answer keys.

Models run through one of three backends, all driven from the same UI:

- **Ollama (direct)** — talks straight to `/api/generate` against a
  local or remote Ollama. Streams tokens natively.
- **vLLM (direct)** — SPECTER spins up a vLLM container *for you*
  (via the host Docker socket), waits for it to be healthy, then
  benchmarks against `/v1/completions` with a model-specific chat
  template (gpt-oss harmony format vs. generic ChatML).
- **LiteLLM (proxy)** — point it at any OpenAI-compatible router and
  benchmark whatever models it exposes; per-model `vllm/*` names get
  auto-routed and the matching vLLM container is brought up on demand.

Everything is observable in real time: live tokens/sec and TTFT charts,
a GPU memory + utilisation strip, a per-prompt streaming rate counter,
and a leaderboard that persists in Redis across runs. Click a leaderboard
row and a full **per-prompt report** opens with the model's actual
output (including separated *thinking* tokens for reasoning models),
category-level breakdowns, and a head-to-head comparison chart.

There are 40 curated prompts shipped in four sets (intelligence,
tactical, coding, reasoning) with editable answer keys, so you can
rerun the same eval against different models / quantisations / engines
and watch the leaderboard reorder itself.

---

## Capabilities (at a glance)

- Three benchmark backends in one UI: **Ollama**, **vLLM**, **LiteLLM**.
- Streaming-aware measurement: TTFT from the first token, sustained
  tokens/sec across the whole generation.
- Live GPU monitoring (memory, utilisation, power) at 1 Hz, broadcast
  over WebSocket to the browser.
- Unified-memory GPU support (Grace Blackwell, Apple Silicon-style
  shared memory) — falls back to Ollama / vLLM API queries when
  `nvidia-smi` reports `[N/A]` for memory.
- 40 prompts in 4 curated sets (intelligence, tactical, coding,
  reasoning), with in-UI editing, add/delete, and pluggable scoring
  (substring-match correctness and entity-extraction accuracy).
- Per-prompt detail report: model output, *thinking* tokens for
  reasoning models, category-level performance, head-to-head
  leaderboard comparison.
- Persistent leaderboard in Redis: latest run per (model, backend) pair,
  sorted by tokens/sec, with category drill-downs.
- vLLM container lifecycle automation: start / switch / stop a vLLM
  container per model from the UI, with automatic recovery to the last
  known-good model on a failed load.
- Optional model isolation: before each run, unload all *other* Ollama
  models from VRAM so measurements aren't contaminated.
- Optional Langfuse tracing: every generation logged with prompt,
  response, scores, TTFT, and thinking-token counts.
- Bundled Redis + optional Caddy HTTPS proxy.

---

## The benchmarks

Each prompt set lives in `prompts/<name>.json` and contains 10 prompts
with an `id`, `name`, `prompt` text, `category`, and (optionally) an
`expected_answer` (for substring-match scoring) or `expected_entities`
(for table-row counting on extraction tasks).

| Set | What it stress-tests | Sample prompts |
|---|---|---|
| **intelligence** (10) | Long-context reading + structured output: entity extraction, timeline reconstruction, threat assessment, network analysis, code-word decryption, SITREP generation, pattern-of-life analysis, source-reliability rating, INTSUM writing, counterintelligence assessment. | `intel-01` Entity Extraction (12 named entities expected); `intel-06` SITREP from grid-coordinate field obs; `intel-09` 300-word INTSUM. |
| **tactical** (10) | Military doctrine + rule-following: course-of-action development, OPORD drafting, ROE analysis, BDA reporting, convoy planning, fire-support planning, MEDEVAC requests, etc. | `tac-01` 3 COAs for a contested river crossing; `tac-02` company-level OPORD; `tac-04` BDA for a 2x JDAM strike. |
| **coding** (10) | Real-world programming: binary search w/ tests, token-bucket rate limiter, Apache log parser, async port scanner, Postgres query optimisation, and more — most have an `expected_answer` substring (e.g. `def binary_search`) for crude correctness scoring. | `code-01` Binary search; `code-04` Async port scanner; `code-05` SQL query optimisation across 50M-row tables. |
| **reasoning** (10) | Multi-step reasoning with verifiable answers: Einstein's 5-houses puzzle (answer: German), trains-meeting word problem, probability, causal reasoning, game-theory Nash equilibrium, Fermi estimation, logical fallacies, cryptarithmetic (`SEND + MORE = MONEY`), ethical-framework analysis, systems-thinking. | `reason-01` Logic Puzzle; `reason-02` Math Word Problem; `reason-08` Cryptarithmetic. |

Scoring (per prompt, per model):

- **`tokens_per_sec`** — `total_tokens / total_time` from streaming.
- **`ttft_ms`** — wall-clock from request send to the first non-empty
  token.
- **`thinking_tokens`** — for reasoning models that emit a separate
  thinking channel (Ollama's `thinking` field, OpenAI's
  `reasoning_content`, gpt-oss's `analysis` channel), tracked
  separately from response tokens.
- **`scores.correct`** — `True` if `expected_answer` (case-insensitive)
  appears anywhere in the response.
- **`scores.entity_count` / `entity_pct`** — counts table rows in the
  response and compares to `expected_entities`.
- **`scores.response_length` / `word_count`** — always recorded.

Aggregates are computed per (model, run) and broken out by category, so
you can see e.g. "model X is 2× faster but 30% less accurate on
reasoning prompts than model Y."

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). The Dockerfile pins the
**aarch64** Docker static binary (downloaded inside the image so SPECTER
can drive its own vLLM container via the host Docker socket); on
**x86_64** hosts you'll need to swap that URL — see the Configuration
section.

The demo's *workload* is portable to any Linux + NVIDIA host: only the
bundled Ollama service requires a GPU. SPECTER itself is a Python
FastAPI app and runs CPU-only.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack pass-through GPU support — bundled Ollama / vLLM won't work. SPECTER itself would, but there's not much point. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, ≥ 8 GB VRAM | Only required if you use the bundled Ollama (`--profile local-models`) or the in-UI vLLM lifecycle. SPECTER + Redis are CPU-only. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to bundled Ollama / vLLM. |
| Disk | ~10 GB image + 2–40 GB per Ollama model + 5–65 GB per vLLM model | vLLM's `gpt-oss-120b` is 65 GB; `llama3.2:3b` on Ollama is ~2 GB. |
| RAM | 16 GB recommended | SPECTER container is capped at 2 GB; the rest is for Ollama / vLLM. |
| Ollama | A reachable Ollama (or any OpenAI-compatible endpoint) | Set `OLLAMA_BASE_URL` in `.env`, **or** start with `--profile local-models` to bring up the bundled GPU Ollama. |
| HF token | Optional | Only for gated vLLM models (e.g. Llama 3.2 instruct). |
| API key | None | Everything runs locally. |

---

## Installation (step-by-step)

Skip to step 4 if you already have Docker + the NVIDIA Container Toolkit
working on this host.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this fails, **fix your NVIDIA driver before continuing** —
the bundled Ollama / vLLM won't start without it.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/specter.git
cd specter
```

### 5. Create the environment file

```bash
cp .env.example .env
```

Two variables are **required** before the stack will start:

| Variable | Set to | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://demo-ollama:11434/v1` if you'll use the bundled Ollama (`--profile local-models`); `http://host.docker.internal:11434/v1` to talk to the host's Ollama; or any OpenAI-compatible endpoint URL. | The trailing `/v1` is optional — SPECTER strips it if present and re-adds it for OpenAI-format calls. |
| `LLM_MODEL` | The default model name to populate the UI dropdown (e.g. `llama3.2:3b`). | Must already be pulled in the Ollama you're pointing at (`ollama list`). |

### 6. Build and start

Default — talks to whatever `OLLAMA_BASE_URL` points at, no GPU
container started by SPECTER:

```bash
docker compose up -d --build
```

To also bring up the bundled GPU Ollama service:

```bash
docker compose --profile local-models up -d --build
docker exec demo-ollama ollama pull llama3.2:3b      # ~2 GB
docker exec demo-ollama ollama pull gpt-oss:20b      # ~12 GB
```

The first build takes 1–3 minutes. The bundled Ollama starts empty —
you must `ollama pull` each model you want to benchmark.

### 7. Verify it's healthy

```bash
docker compose ps
# demo-specter and demo-redis should show "healthy" within ~30 s
# demo-ollama (if started) shows "healthy" within ~60 s

curl -s http://localhost:8080/health
```

Expected:

```json
{"status": "ok", "service": "specter"}
```

### 8. Open the UI

> **Important:** the UI is served under a `/specter/` path prefix
> (the static HTML uses `<base href="/specter/">` and every API call in
> the bundled JavaScript hits `/specter/api/...`). Open it at:

- **Workbench:** <http://localhost:8080/specter/>

Opening `http://localhost:8080/` (no trailing path) will load the HTML
but the JS will 404 on every API call and the page will sit blank.
Either always use the `/specter/` URL, or put a parent reverse proxy in
front that strips `/specter/` and forwards to `specter:8080/`.

### 9. (Optional) Tail the logs

```bash
docker compose logs -f specter
```

You should see `SPECTER ready` once Redis is connected and the GPU
monitor is running.

---

## Configuration

All variables can be set in `.env` or exported in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `OLLAMA_BASE_URL` | _(required)_ | OpenAI-compatible endpoint. Use `http://demo-ollama:11434/v1` for the bundled Ollama or `http://host.docker.internal:11434/v1` for the host's. Trailing `/v1` is optional. |
| `LLM_MODEL` | _(required)_ | Default model name shown in the UI. Must already exist in the Ollama you pointed at. |
| `APP_PORT` | `8080` | Browser-facing port. |
| `OLLAMA_HOST_PORT` | `11434` | Host port the bundled Ollama is exposed on. Rare to change. |
| `REDIS_HOST_PORT` | `6379` | Host port the bundled Redis is exposed on. |
| `REDIS_URL` | `redis://demo-redis:6379/15` | Connection string used by SPECTER. Override when you BYO Redis. |
| `VLLM_BASE_URL` | _(empty)_ | OpenAI-compatible vLLM endpoint. SPECTER manages this container itself when set; leave empty if you don't need vLLM benchmarks. Default at runtime is `http://demo-vllm:8000`. |
| `VLLM_COMPOSE_DIR` | `/vllm-compose` | Reserved for an external vLLM compose dir; not currently used by the in-UI lifecycle. |
| `HF_TOKEN` | _(empty)_ | Hugging Face token, passed into the vLLM container for downloading gated models (e.g. `meta-llama/Llama-3.2-3B-Instruct`). |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | _(empty)_ | If all three are set, every benchmark generation is logged to your Langfuse instance with prompt, response, scores, TTFT, and thinking-token counts. |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port. |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port. |

### x86_64 build override

The Dockerfile downloads the **aarch64** Docker static binary so SPECTER
can drive its own vLLM container. On x86_64 hosts, edit the
`docker.com/linux/static/.../aarch64/...` URL in the Dockerfile to the
matching `x86_64` artifact before building, e.g.:

```
https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz
```

If you're not using the in-UI vLLM lifecycle, you don't strictly need
the docker CLI inside the container at all — but the Dockerfile bakes
it in unconditionally.

---

## Live controls (in the browser)

The workbench has two tabs (left panel) and a metrics panel (right).

**Configuration (always visible):**

- **Backend** — Ollama (Direct), LiteLLM (Proxy), or vLLM (Direct).
- **Runs per prompt** — 1, 2, or 3 (each prompt is run N times, all
  results captured).

**Benchmark tab:**

- **Models list** — auto-populated from `OLLAMA_BASE_URL`'s `/api/tags`.
  Multi-select. Each row shows arch type (Dense / MoE), parameter size,
  quant level, on-disk size, and (if loaded) live VRAM use.
- **Refresh / Load / Unload** — refresh the list, force-load selected
  models into VRAM (Ollama `keep_alive=10m`), or evict them
  (`keep_alive=0`).
- **Prompt sets** — multi-select intelligence / tactical / coding /
  reasoning.
- **RUN BENCHMARK / STOP** — start the iteration; stop at any time
  (current prompt finishes, then the run is marked incomplete and
  excluded from the leaderboard).

**Prompts tab:**

- **Prompt set selector** — view and edit prompts in any set.
- **Add / Edit / Delete** — full CRUD with name, category, prompt text,
  expected answer (substring), and expected entity count. Changes are
  written back to the JSON files on disk.

**Metrics panel (right):**

- **Live charts** — tokens/sec, GPU memory, TTFT, GPU utilisation
  (rolling window).
- **Leaderboard** — sorted by avg tokens/sec; one row per (model,
  backend) pair, showing arch, engine, tok/s, TTFT, accuracy, prompts
  completed. Click a row → opens a **per-prompt report modal** with:
  - Summary cards (avg tok/s, TTFT, accuracy, total tokens).
  - Category-level performance bar chart.
  - Head-to-head leaderboard comparison chart.
  - Per-prompt table with the model's full response, *thinking* output,
    and score — click each row to expand.
- **Run Results** — live per-prompt rows for the active benchmark.

---

## External services (BYO)

If you'd rather use your own Redis (e.g. a managed instance), set
`REDIS_URL` in `.env` and start with the BYO override:

```bash
docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

The override runs the bundled Redis with `replicas: 0` (so it doesn't
start) and removes the `depends_on: redis` from the SPECTER service.
The bundled Ollama is already gated under `--profile local-models`, so
it never starts unless explicitly requested.

| Variable | Example |
|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` (or any OpenAI-compatible URL) |
| `REDIS_URL` | `redis://redis.example.com:6379/15` |

Redis is used **only** for the persistent leaderboard / run history.
SPECTER will run without Redis — the lifespan logs a warning and
`history` / `leaderboard` endpoints return empty.

---

## vLLM lifecycle (advanced)

The vLLM backend is unique: SPECTER **manages a vLLM container itself**
via the mounted host Docker socket, recreating it whenever the user
switches model in the UI. The full list of supported models lives in
`server.py:VLLM_MODELS` and includes:

- `openai/gpt-oss-20b` (MoE, MXFP4, ~13 GB)
- `openai/gpt-oss-120b` (MoE, MXFP4, ~65 GB)
- `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (MoE, ~18 GB)
- `Qwen/Qwen3-32B-AWQ` (Dense, ~20 GB)
- `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (~5 GB)
- `meta-llama/Llama-3.2-3B-Instruct` (gated — needs `HF_TOKEN`)
- `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` (~17 GB)

Things to know before enabling vLLM:

- The vLLM container is launched with `--network demo-network`. **You
  must create that Docker network ahead of time** if it doesn't already
  exist:
  `docker network create demo-network`
  and attach `demo-specter` to it (or run the whole stack in a compose
  project that uses `demo-network` as its default).
- The model cache path is **hardcoded** to
  `/home/ameinecke/demo_center/data/vllm-models` on the host — change
  this in `server.py:_vllm_start_container` for your environment, or
  symlink/bind-mount your own cache there.
- A tiktoken cache must exist at
  `/home/ameinecke/demo_center/data/vllm-models/tiktoken-cache/raw` for
  gpt-oss models to load offline.
- Switching models stops + recreates the container. If a load fails,
  SPECTER automatically rolls back to the previously-working model.

---

## Optional HTTPS reverse proxy

Caddy is bundled as an opt-in profile. It auto-provisions Let's
Encrypt certs when `DEMO_HOSTNAME` is a real DNS name pointing at this
host:

```bash
DEMO_HOSTNAME=specter.example.com docker compose --profile proxy up -d
```

Note: the bundled `Caddyfile` does a plain `reverse_proxy specter:8080`
without rewriting the path, so the UI is still reachable at
`https://specter.example.com/specter/`. If you want a clean root URL,
add a `handle_path /specter/* { ... }` block or front the demo with
your own Traefik / nginx that strips the prefix.

For local testing keep `DEMO_HOSTNAME=localhost` — Caddy issues a
self-signed cert.

---

## Authentication

SPECTER runs **without authentication** by default. There is one extra
risk worth being aware of: the SPECTER container mounts the **host
Docker socket** (read-only by default in `docker-compose.yml`, but
SPECTER also issues `docker run` and `docker stop` against it for the
vLLM lifecycle — change to read-write if you actually want vLLM control
to work). A user with HTTP access to SPECTER could plausibly use the
vLLM endpoints to influence container creation. Keep this demo on
trusted networks or behind one of:

- **Caddy basic auth** — add a `basic_auth` block to the Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

If you don't need vLLM benchmarks, comment out the
`/var/run/docker.sock` mount in `docker-compose.yml`.

---

## Architecture (file map)

| File | Purpose |
|---|---|
| `server.py` | FastAPI app: REST + WebSocket endpoints, GPU monitor loop, benchmark runner, vLLM container lifecycle, Langfuse tracing. |
| `benchmarks.py` | Prompt-set CRUD (load / save / add / update / delete), `BenchmarkResult` dataclass, response scoring (`expected_answer` substring + `expected_entities` table-row counting). |
| `prompts/intelligence.json` | 10 SIGINT/HUMINT/GEOINT-style analysis + generation prompts. |
| `prompts/tactical.json` | 10 doctrinal prompts (COA, OPORD, ROE, BDA, MEDEVAC, fire-support, etc.). |
| `prompts/coding.json` | 10 implementation + optimisation prompts. |
| `prompts/reasoning.json` | 10 multi-step reasoning prompts with verifiable answers. |
| `static/index.html` | Workbench UI (single page, two tabs, metrics panel, report modal, prompt editor modal). Uses `<base href="/specter/">`. |
| `static/js/app.js` | Frontend logic: model discovery, benchmark runs, vLLM lifecycle, leaderboard, report modal. All API calls hardcoded to `/specter/`. |
| `static/js/charts.js` | Lightweight canvas chart helpers (no Chart.js dependency). |
| `static/css/specter.css` | Cyber/HUD-styled CSS. |
| `Dockerfile` | `python:3.11-slim` + Docker static binary (aarch64) + `requirements.txt`. |
| `docker-compose.yml` | SPECTER + Redis (always); Ollama (`--profile local-models`); Caddy (`--profile proxy`). |
| `docker-compose.byo.yml` | Override that disables the bundled Redis. |
| `Caddyfile` | Simple `reverse_proxy specter:8080` (no path rewrite). |

---

## Troubleshooting

- **Page loads but nothing shows / "Loading models..." forever** — you
  opened `http://localhost:8080/` instead of
  `http://localhost:8080/specter/`. The frontend hardcodes the
  `/specter/` prefix on every API call. Open the DevTools network tab
  and confirm the requests are 404ing on `/api/...`; switch to the
  `/specter/` URL and they'll resolve.
- **`OLLAMA_BASE_URL` unreachable** — from inside the container, run
  `docker exec demo-specter sh -c 'curl -s ${OLLAMA_BASE_URL%/v1}/api/tags'`.
  If you set `http://host.docker.internal:...` on Linux, make sure the
  compose service has `extra_hosts: ["host.docker.internal:host-gateway"]`
  or switch to a LAN IP / `172.17.0.1`.
- **GPU memory shows `0` / `[N/A]` on a Grace Blackwell or unified-memory
  GPU** — expected. SPECTER falls back to summing Ollama's `/api/ps`
  `size_vram` field plus the active vLLM model's known size. Make sure
  your benchmark target is actually loaded (Load button or run a
  benchmark) before reading the chart.
- **GPU not visible (with bundled Ollama)** — confirm `nvidia-smi`
  works on the host and that
  `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi`
  works too. If not, fix the NVIDIA Container Toolkit before going
  further.
- **Ollama model missing** — `docker exec demo-ollama ollama pull <model>`
  for each model you want to benchmark. The bundled Ollama starts
  empty.
- **vLLM benchmark fails immediately** — most likely causes:
  1. `demo-network` Docker network doesn't exist —
     `docker network create demo-network` and connect `demo-specter` to
     it.
  2. The hardcoded host path `/home/ameinecke/demo_center/data/vllm-models`
     doesn't exist on your host. Either edit `server.py` to your real
     path or symlink it.
  3. Gated model — set `HF_TOKEN` in `.env` and recreate the SPECTER
     container.
  Look at `docker logs demo-vllm` after a failed switch — SPECTER
  surfaces the last 300 chars of vLLM's stderr in the UI status banner.
- **Benchmark "completed" but excluded from leaderboard** — the run was
  cancelled. Cancelled runs are stored with `summary.completed = false`
  and intentionally skipped by the leaderboard so they don't overwrite
  finished results. The full per-prompt data is still available in
  history via `GET /api/history` and `GET /api/run/{id}`.
- **`accuracy` is `null` in the leaderboard** — none of the prompts you
  ran had an `expected_answer` field set. Add one in the Prompts tab to
  enable substring-match scoring.
- **Port collision** — change `APP_PORT` (or `OLLAMA_HOST_PORT` /
  `REDIS_HOST_PORT`) in `.env`.
- **`mem_limit: 2g` OOMs the SPECTER container** — increase or remove
  the limit in `docker-compose.yml`. With Langfuse + a long history
  this can become tight.

---

## FAQ

**Q: Can I use a CPU-only host?** SPECTER + Redis are CPU-only, but
Ollama / vLLM aren't useful without a GPU. Point `OLLAMA_BASE_URL` at a
remote GPU host running Ollama and the demo works fine on a laptop.

**Q: Does it work with non-Ollama OpenAI-compatible endpoints (vLLM
standalone, LiteLLM, llama.cpp, OpenRouter, etc.)?** Yes — the LiteLLM
backend just streams `/v1/chat/completions` against whatever
`OLLAMA_BASE_URL` you set. Model discovery (the left-panel list) does
require the Ollama-specific `/api/tags` endpoint though, so for pure
non-Ollama setups you may need to manually type model names.

**Q: How accurate is the scoring?** Substring match — crude but
deterministic. It's enough to catch wildly-wrong outputs and rank
models on simple verifiable tasks (`reason-01` correct answer is
`German`, `reason-08` is `9567 + 1085 = 10652`). For nuanced eval, hook
Langfuse up and grade externally.

**Q: How do I reset the leaderboard?** `docker compose down -v` (the
`-v` removes the named Redis volume) and restart. Or selectively delete
runs via `DELETE /api/run/{run_id}`.

**Q: Why two URL paths (`/specter/` for the UI, `/` for `/health`)?**
The frontend was originally built to live behind a parent demo router
that strips `/specter/`. The backend serves API + health at root; the
frontend assets reference `/specter/` because that's what the parent
router exposes externally. When running standalone, just use
`http://localhost:8080/specter/` for the UI and `http://localhost:8080/health`
for the health probe.

---

## Credits

Built by Andrew Meinecke.
