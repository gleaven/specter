"""SPECTER — Edge Inference Optimization Workbench."""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from benchmarks import (
    BenchmarkResult,
    add_prompt,
    delete_prompt,
    list_prompt_sets,
    load_prompt_set,
    score_response,
    update_prompt,
)

# ── Configuration ──────────────────────────────────────────────
# OLLAMA_BASE_URL may be passed with or without a trailing /v1 (the
# OpenAI-compatible suffix). Strip it so the native /api/* calls in
# this file work either way.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://demo-ollama:11434").rstrip("/")
if OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL[:-3]
# LITELLM_BASE_URL is kept as a name but is now whatever OpenAI-compatible
# endpoint the user pointed OLLAMA_BASE_URL at (default: bundled demo-ollama).
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", f"{OLLAMA_BASE_URL}/v1").rstrip("/")
if LITELLM_BASE_URL.endswith("/v1/v1"):
    LITELLM_BASE_URL = LITELLM_BASE_URL[:-3]
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "not-used")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://demo-vllm:8000")
VLLM_COMPOSE_DIR = os.environ.get("VLLM_COMPOSE_DIR", "/vllm-compose")
# Host path of the vLLM models cache. Required only if you use the
# vLLM benchmarking flow — specter spawns sibling vLLM containers via
# the docker socket and bind-mounts this path into them as
# /root/.cache/huggingface, so it must be a HOST path the docker
# daemon can resolve.
VLLM_MODELS_DIR = os.environ.get("VLLM_MODELS_DIR", "").rstrip("/")
REDIS_URL = os.environ.get("REDIS_URL", "redis://demo-redis:6379/18")
SERVICEROUTER_URL = os.environ.get("SERVICEROUTER_URL", "http://demo-servicerouter:8080")

# Known vLLM-compatible models with GPU memory requirements
# 4-bit quantized models for fair comparison with Ollama's Q4_K_M
VLLM_MODELS = {
    "openai/gpt-oss-20b": {"gpu_mem": "0.50", "max_model_len": "32768", "arch_type": "MoE", "parameter_size": "21B (3.6B active)", "quant": "MXFP4", "size_gb": 13, "litellm_name": "vllm/gpt-oss-20b"},
    "openai/gpt-oss-120b": {"gpu_mem": "0.70", "max_model_len": "16384", "arch_type": "MoE", "parameter_size": "117B (5.1B active)", "quant": "MXFP4", "size_gb": 65, "litellm_name": "vllm/gpt-oss-120b"},
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4": {"gpu_mem": "0.35", "max_model_len": "4096", "arch_type": "MoE", "parameter_size": "30B", "quant": "GPTQ-4bit", "size_gb": 18, "litellm_name": "vllm/qwen3-30b-a3b"},
    "Qwen/Qwen3-32B-AWQ": {"gpu_mem": "0.35", "max_model_len": "4096", "arch_type": "Dense", "parameter_size": "32B", "quant": "AWQ-4bit", "size_gb": 20, "litellm_name": "vllm/qwen3-32b"},
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": {"gpu_mem": "0.20", "max_model_len": "4096", "arch_type": "Dense", "parameter_size": "8B", "quant": "AWQ-4bit", "size_gb": 5, "litellm_name": "vllm/llama-3.1-8b"},
    "meta-llama/Llama-3.2-3B-Instruct": {"gpu_mem": "0.15", "max_model_len": "4096", "arch_type": "Dense", "parameter_size": "3.2B", "quant": "FP16", "size_gb": 6, "gated": True, "litellm_name": "vllm/llama-3.2-3b"},
    "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g": {"gpu_mem": "0.35", "max_model_len": "4096", "arch_type": "Dense", "parameter_size": "27B", "quant": "GPTQ-4bit", "size_gb": 17, "litellm_name": "vllm/gemma-3-27b"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("specter")

# ── Langfuse ──────────────────────────────────────────────────
_langfuse = None
try:
    from langfuse import Langfuse
    _lf_host = os.environ.get("LANGFUSE_HOST")
    _lf_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    _lf_sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if _lf_host and _lf_pk and _lf_sk:
        _langfuse = Langfuse(public_key=_lf_pk, secret_key=_lf_sk, host=_lf_host)
        logger.info("Langfuse tracing enabled")
    else:
        logger.info("Langfuse env vars not set — tracing disabled")
except ImportError:
    logger.info("Langfuse SDK not installed — tracing disabled")

# ── Global State ───────────────────────────────────────────────
_redis_client: redis.Redis | None = None
_ws_clients: set[WebSocket] = set()
_active_benchmark: dict | None = None  # {run_id, cancel}
_gpu_task: asyncio.Task | None = None
_nvidia_smi_available: bool | None = None


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client, _gpu_task
    logger.info("SPECTER starting up...")

    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info(f"Redis connected ({REDIS_URL.rsplit('/', 1)[-1]})")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e} — running without persistence")
        _redis_client = None

    _gpu_task = asyncio.create_task(_gpu_monitor_loop())
    logger.info("SPECTER ready")
    yield

    logger.info("SPECTER shutting down...")
    if _gpu_task:
        _gpu_task.cancel()
        try:
            await _gpu_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="SPECTER", lifespan=lifespan)


# ── Broadcast Helpers ──────────────────────────────────────────
async def broadcast(msg: dict):
    """Send JSON message to all connected WebSocket clients."""
    text = json.dumps(msg)
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── GPU Monitor ────────────────────────────────────────────────
async def _gpu_monitor_loop():
    """Poll GPU stats every second and broadcast to WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(1)
            if not _ws_clients:
                continue
            stats = await _get_gpu_stats()
            if stats:
                await broadcast({"type": "gpu", **stats})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"GPU monitor error: {e}")
            await asyncio.sleep(5)


async def _check_nvidia_smi() -> bool:
    """Check if nvidia-smi is available (cached)."""
    global _nvidia_smi_available
    if _nvidia_smi_available is not None:
        return _nvidia_smi_available
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5)
        _nvidia_smi_available = proc.returncode == 0
    except Exception:
        _nvidia_smi_available = False
    logger.info(f"nvidia-smi available: {_nvidia_smi_available}")
    return _nvidia_smi_available


async def _get_gpu_stats() -> dict | None:
    """Get GPU stats from nvidia-smi + Ollama API (handles unified memory GPUs)."""
    stats = {}

    # nvidia-smi for utilization and power (may return [N/A] for memory on unified GPUs)
    if await _check_nvidia_smi():
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            line = stdout.decode().strip()
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    # Parse values, treating [N/A] as unavailable
                    def parse_val(s):
                        try:
                            return float(s)
                        except (ValueError, TypeError):
                            return -1

                    mem_used = parse_val(parts[0])
                    mem_total = parse_val(parts[1])
                    gpu_util = parse_val(parts[2])
                    power = parse_val(parts[3])

                    if gpu_util >= 0:
                        stats["gpu_util_pct"] = gpu_util
                    if power >= 0:
                        stats["power_w"] = power
                    if mem_used >= 0:
                        stats["memory_used_mb"] = mem_used
                    if mem_total >= 0:
                        stats["memory_total_mb"] = mem_total
        except Exception:
            pass

    # Supplement with Ollama + vLLM APIs for memory (essential for unified memory GPUs)
    if "memory_used_mb" not in stats:
        total_vram_bytes = 0
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Ollama loaded models
                try:
                    r = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
                    if r.status_code == 200:
                        for m in r.json().get("models", []):
                            total_vram_bytes += m.get("size_vram", 0)
                except Exception:
                    pass
                # vLLM: check if running and estimate from model config
                try:
                    r = await client.get(f"{VLLM_BASE_URL}/v1/models")
                    if r.status_code == 200:
                        vllm_models = r.json().get("data", [])
                        if vllm_models:
                            active_id = vllm_models[0].get("id", "")
                            meta = VLLM_MODELS.get(active_id, {})
                            vllm_gb = meta.get("size_gb", 0)
                            if vllm_gb:
                                total_vram_bytes += int(vllm_gb * 1024**3)
                except Exception:
                    pass
        except Exception:
            pass
        if total_vram_bytes > 0:
            stats["memory_used_mb"] = round(total_vram_bytes / (1024**2), 1)

    # Defaults for missing fields
    stats.setdefault("memory_used_mb", 0)
    stats.setdefault("memory_total_mb", 0)
    stats.setdefault("gpu_util_pct", -1)
    stats.setdefault("power_w", -1)

    return stats if (stats.get("gpu_util_pct", -1) >= 0 or stats.get("memory_used_mb", 0) > 0) else None


# ── Health Check ───────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "specter"}


# ── Model Discovery ───────────────────────────────────────────
@app.get("/api/models")
async def api_models():
    """List all Ollama models with metadata."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
            models_data = r.json().get("models", [])

            try:
                ps_r = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
                ps_data = ps_r.json().get("models", [])
                loaded_map = {m["name"]: m for m in ps_data}
            except Exception:
                loaded_map = {}

            # Known MoE architecture families
            MOE_FAMILIES = {"qwen3moe", "gptoss", "deepseek2", "mixtral", "dbrx", "arctic"}

            enriched = []
            for m in models_data:
                model_name = m.get("name", "")
                family = m.get("details", {}).get("family", "")
                info = {
                    "name": model_name,
                    "size_bytes": m.get("size", 0),
                    "size_gb": round(m.get("size", 0) / (1024**3), 1),
                    "modified_at": m.get("modified_at", ""),
                    "family": family,
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    "quantization_level": m.get("details", {}).get("quantization_level", ""),
                    "format": m.get("details", {}).get("format", ""),
                    "arch_type": "MoE" if family.lower() in MOE_FAMILIES else "Dense",
                }

                if model_name in loaded_map:
                    loaded = loaded_map[model_name]
                    info["loaded"] = True
                    info["vram_bytes"] = loaded.get("size_vram", 0)
                    info["vram_gb"] = round(loaded.get("size_vram", 0) / (1024**3), 1)
                else:
                    info["loaded"] = False

                enriched.append(info)

            return {"models": enriched}
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"Ollama unavailable: {e}"}, status_code=503)


# ── GPU Stats API ──────────────────────────────────────────────
@app.get("/api/gpu")
async def api_gpu():
    """Current GPU stats."""
    stats = await _get_gpu_stats()
    if stats:
        return stats
    return JSONResponse({"error": "GPU stats unavailable"}, status_code=503)


# ── Model Load / Unload ──────────────────────────────────────
@app.post("/api/models/load")
async def api_load_models(body: dict):
    """Load models into GPU memory via Ollama."""
    models = body.get("models", [])
    if not models:
        return JSONResponse({"error": "No models specified"}, status_code=400)
    results = {}
    async with httpx.AsyncClient(timeout=120.0) as client:
        for name in models:
            try:
                r = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": name, "prompt": "", "stream": False, "keep_alive": "10m"},
                    timeout=120.0,
                )
                results[name] = "loaded" if r.status_code == 200 else f"error ({r.status_code})"
            except Exception as e:
                results[name] = f"error: {e}"
    return {"results": results}


@app.post("/api/models/unload")
async def api_unload_models(body: dict):
    """Unload models from GPU memory via Ollama."""
    models = body.get("models", [])
    if not models:
        return JSONResponse({"error": "No models specified"}, status_code=400)
    results = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for name in models:
            try:
                r = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": name, "keep_alive": 0, "stream": False},
                    timeout=30.0,
                )
                results[name] = "unloaded" if r.status_code == 200 else f"error ({r.status_code})"
            except Exception as e:
                results[name] = f"error: {e}"
    return {"results": results}


# ── vLLM Management ──────────────────────────────────────────
@app.get("/api/vllm/models")
async def api_vllm_models():
    """List available vLLM models and current status."""
    # Get currently loaded model from vLLM
    active_model = None
    vllm_up = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_BASE_URL}/v1/models")
            if r.status_code == 200:
                vllm_up = True
                models = r.json().get("data", [])
                if models:
                    active_model = models[0].get("id")
    except Exception:
        pass

    model_list = []
    for name, meta in VLLM_MODELS.items():
        model_list.append({
            "name": name,
            "active": name == active_model,
            **meta,
        })

    return {"models": model_list, "active_model": active_model, "vllm_up": vllm_up}


_vllm_last_working_model: str | None = None  # Track last good model for recovery


async def _vllm_start_container(model: str, meta: dict) -> tuple[bool, str]:
    """Start a vLLM container with the given model. Returns (success, error_msg)."""
    if not VLLM_MODELS_DIR:
        return False, (
            "VLLM_MODELS_DIR is not set. Add it to .env pointing at the host "
            "path of your HuggingFace models cache (e.g. ~/data/vllm-models)."
        )
    # Remove any existing container
    for cmd in [["docker", "stop", "-t", "10", "demo-vllm"], ["docker", "rm", "-f", "demo-vllm"]]:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

    run_cmd = [
        "docker", "run", "-d",
        "--name", "demo-vllm",
        "--network", "demo-network",
        "--shm-size", "16g",
        "--gpus", "all",
        "-v", f"{VLLM_MODELS_DIR}:/root/.cache/huggingface",
        "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "-e", f"HUGGING_FACE_HUB_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "-e", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "-e", "TIKTOKEN_ENCODINGS_BASE=/tiktoken-files/",
        "-e", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1",
        "-v", f"{VLLM_MODELS_DIR}/tiktoken-cache/raw:/tiktoken-files:ro",
        "vllm/vllm-openai:latest",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", meta.get("max_model_len", "4096"),
        "--gpu-memory-utilization", meta["gpu_mem"],
        "--dtype", "auto",
        "--trust-remote-code",
        "--disable-frontend-multiprocessing",
    ]

    proc = await asyncio.create_subprocess_exec(
        *run_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        return False, f"docker run failed: {stderr.decode()[:200]}"

    # Poll for readiness — also detect early crashes
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _ in range(60):  # up to 5 minutes
            await asyncio.sleep(5)
            # Check if container is still running
            check = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", "demo-vllm",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await check.communicate()
            if out.decode().strip() != "true":
                # Container crashed — get last logs
                log_proc = await asyncio.create_subprocess_exec(
                    "docker", "logs", "--tail", "5", "demo-vllm",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                log_out, log_err = await log_proc.communicate()
                error_lines = (log_out.decode() + log_err.decode()).strip()[-300:]
                return False, f"Container crashed: {error_lines}"

            try:
                r = await client.get(f"{VLLM_BASE_URL}/health")
                if r.status_code == 200:
                    return True, ""
            except Exception:
                pass

    return False, "Not healthy after 5 minutes"


@app.post("/api/vllm/switch")
async def api_vllm_switch(body: dict):
    """Switch the active vLLM model by recreating the container."""
    global _vllm_last_working_model

    model = body.get("model", "")
    if model not in VLLM_MODELS:
        return JSONResponse({"error": f"Unknown model: {model}"}, status_code=400)

    meta = VLLM_MODELS[model]

    # Block models known to be broken
    if meta.get("broken"):
        return JSONResponse({"error": f"{model} is currently unsupported in vLLM (see notes)"}, status_code=400)

    try:
        logger.info(f"Switching vLLM to {model} (gpu_mem={meta['gpu_mem']})")
        await broadcast({"type": "status_msg", "msg": f"Switching vLLM to {model}..."})

        # Remember previous working model for recovery
        previous_model = _vllm_last_working_model

        success, error = await _vllm_start_container(model, meta)

        if success:
            _vllm_last_working_model = model
            await broadcast({"type": "status_msg", "msg": f"vLLM ready: {model}"})
            return {"status": "ready", "model": model}
        else:
            logger.error(f"vLLM failed to load {model}: {error}")
            await broadcast({"type": "status_msg", "msg": f"Failed: {model}. Recovering..."})

            # Try to recover with previous working model
            if previous_model and previous_model != model:
                prev_meta = VLLM_MODELS.get(previous_model)
                if prev_meta:
                    logger.info(f"Recovering vLLM with previous model: {previous_model}")
                    ok, _ = await _vllm_start_container(previous_model, prev_meta)
                    if ok:
                        _vllm_last_working_model = previous_model
                        await broadcast({"type": "status_msg", "msg": f"Recovered: {previous_model}"})

            return JSONResponse({"error": f"Failed to load {model}: {error}"}, status_code=500)

    except Exception as e:
        logger.error(f"vLLM switch error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/vllm/stop")
async def api_vllm_stop():
    """Stop the vLLM container (unload model)."""
    try:
        logger.info("Stopping vLLM container")
        await broadcast({"type": "status_msg", "msg": "Stopping vLLM..."})
        proc = await asyncio.create_subprocess_exec(
            "docker", "stop", "-t", "10", "demo-vllm",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", "demo-vllm",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
        await broadcast({"type": "status_msg", "msg": "vLLM stopped"})
        return {"status": "stopped"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Prompt Sets ────────────────────────────────────────────────
@app.get("/api/prompts")
async def api_prompts():
    """List available prompt sets."""
    return {"prompt_sets": list_prompt_sets()}


@app.get("/api/prompts/{set_id}/items")
async def api_prompt_items(set_id: str):
    """Get all prompts in a set."""
    try:
        prompts = load_prompt_set(set_id)
        return {"prompts": prompts, "set_id": set_id}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.post("/api/prompts/{set_id}/items")
async def api_add_prompt(set_id: str, body: dict):
    """Add a new prompt to a set."""
    try:
        prompt = add_prompt(set_id, body)
        return {"prompt": prompt}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.put("/api/prompts/{set_id}/items/{prompt_id}")
async def api_update_prompt(set_id: str, prompt_id: str, body: dict):
    """Update an existing prompt."""
    result = update_prompt(set_id, prompt_id, body)
    if result:
        return {"prompt": result}
    return JSONResponse({"error": "Prompt not found"}, status_code=404)


@app.delete("/api/prompts/{set_id}/items/{prompt_id}")
async def api_delete_prompt(set_id: str, prompt_id: str):
    """Delete a prompt from a set."""
    if delete_prompt(set_id, prompt_id):
        return {"status": "deleted"}
    return JSONResponse({"error": "Prompt not found"}, status_code=404)


# ── Benchmark Start ───────────────────────────────────────────
@app.post("/api/benchmark/start")
async def api_benchmark_start(body: dict):
    """Start a benchmark run."""
    global _active_benchmark

    if _active_benchmark:
        return JSONResponse({"error": "Benchmark already running"}, status_code=409)

    models = body.get("models", [])
    prompt_sets = body.get("prompt_sets", [])
    runs_per_prompt = body.get("runs_per_prompt", 1)
    backend = body.get("backend", "ollama")  # "ollama" or "litellm"

    if not models:
        return JSONResponse({"error": "No models specified"}, status_code=400)
    if not prompt_sets:
        return JSONResponse({"error": "No prompt sets specified"}, status_code=400)
    if backend not in ("ollama", "litellm", "vllm"):
        return JSONResponse({"error": "Invalid backend (ollama, litellm, or vllm)"}, status_code=400)

    run_id = str(uuid.uuid4())[:8]
    cancel_event = asyncio.Event()
    _active_benchmark = {"run_id": run_id, "cancel": cancel_event}

    asyncio.create_task(_run_benchmark(run_id, models, prompt_sets, runs_per_prompt, cancel_event, backend))

    return {"run_id": run_id, "status": "started"}


# ── Benchmark Stop ────────────────────────────────────────────
@app.post("/api/benchmark/stop")
async def api_benchmark_stop():
    """Cancel running benchmark."""
    global _active_benchmark
    if _active_benchmark:
        _active_benchmark["cancel"].set()
        return {"status": "stopping"}
    return {"status": "no_benchmark_running"}


# ── Model Isolation ───────────────────────────────────────────
async def _unload_other_models(benchmark_models: list[str]):
    """Unload all models not being benchmarked to free GPU memory."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            if r.status_code != 200:
                return
            loaded = r.json().get("models", [])
            for m in loaded:
                name = m.get("name", "")
                if name not in benchmark_models:
                    logger.info(f"Unloading model {name} to free GPU memory")
                    await broadcast({"type": "status_msg", "msg": f"Unloading {name}..."})
                    try:
                        await client.post(
                            f"{OLLAMA_BASE_URL}/api/generate",
                            json={"model": name, "keep_alive": 0, "stream": False},
                            timeout=30.0,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to unload {name}: {e}")
            await asyncio.sleep(2)
    except Exception as e:
        logger.warning(f"Failed to check loaded models: {e}")


# ── Benchmark Runner ──────────────────────────────────────────
async def _run_benchmark(
    run_id: str,
    models: list[str],
    prompt_set_names: list[str],
    runs_per_prompt: int,
    cancel: asyncio.Event,
    backend: str = "ollama",
):
    """Execute benchmark run: iterate models x prompts, measure streaming performance."""
    global _active_benchmark

    all_results = []
    total_pairs = 0
    completed = 0

    # Count total work
    for ps_name in prompt_set_names:
        try:
            prompts = load_prompt_set(ps_name)
            total_pairs += len(models) * len(prompts) * runs_per_prompt
        except Exception:
            pass

    await broadcast({"type": "benchmark_start", "run_id": run_id, "total": total_pairs})

    # Unload non-benchmark models for clean GPU measurements
    await _unload_other_models(models)

    # Create Langfuse trace for this benchmark run
    lf_trace = None
    if _langfuse:
        try:
            lf_trace = _langfuse.trace(
                name=f"specter-benchmark",
                session_id=run_id,
                metadata={
                    "run_id": run_id,
                    "models": models,
                    "prompt_sets": prompt_set_names,
                    "runs_per_prompt": runs_per_prompt,
                    "total_prompts": total_pairs,
                    "backend": backend,
                },
                tags=["specter", "benchmark", backend],
            )
        except Exception as e:
            logger.warning(f"Langfuse trace creation failed: {e}")

    try:
        for model_name in models:
            if cancel.is_set():
                break

            for ps_name in prompt_set_names:
                if cancel.is_set():
                    break

                try:
                    prompts = load_prompt_set(ps_name)
                except Exception as e:
                    logger.error(f"Failed to load prompt set {ps_name}: {e}")
                    continue

                for prompt_meta in prompts:
                    if cancel.is_set():
                        break

                    for run_num in range(runs_per_prompt):
                        if cancel.is_set():
                            break

                        result = await _benchmark_single(
                            model_name, ps_name, prompt_meta, run_id, backend
                        )
                        rd = result.to_dict()
                        all_results.append(rd)
                        completed += 1

                        # Log generation to Langfuse
                        if lf_trace:
                            try:
                                lf_trace.generation(
                                    name=f"{ps_name}/{prompt_meta['name']}",
                                    model=model_name,
                                    input=prompt_meta["prompt"],
                                    output=rd.get("response_text") or rd.get("thinking_text") or "",
                                    metadata={
                                        "prompt_set": ps_name,
                                        "prompt_id": prompt_meta["id"],
                                        "category": prompt_meta.get("category", ""),
                                        "scores": rd.get("scores"),
                                        "ttft_ms": rd["ttft_ms"],
                                        "thinking_tokens": rd.get("thinking_tokens", 0),
                                    },
                                    usage={
                                        "total_tokens": rd["total_tokens"],
                                    },
                                    level="ERROR" if rd.get("error") else "DEFAULT",
                                    status_message=rd.get("error"),
                                )
                            except Exception as e:
                                logger.warning(f"Langfuse generation log failed: {e}")

                        await broadcast({
                            "type": "benchmark_result",
                            "run_id": run_id,
                            "result": rd,
                            "completed": completed,
                            "total": total_pairs,
                        })

        # Compute summary — mark whether the run completed fully or was cancelled
        was_cancelled = cancel.is_set()
        summary = _compute_summary(run_id, all_results)
        summary["completed"] = not was_cancelled
        summary["completed_count"] = completed
        summary["total_count"] = total_pairs
        summary["backend"] = backend
        # For LiteLLM, detect which engine it routed to from model names
        if backend == "litellm":
            has_vllm = any(m.startswith("vllm/") for m in models)
            summary["engine"] = "vllm" if has_vllm else "ollama"

        # Store in Redis
        if _redis_client:
            try:
                _redis_client.set(
                    f"specter:result:{run_id}",
                    json.dumps({"run_id": run_id, "summary": summary, "results": all_results}),
                )
            except Exception as e:
                logger.warning(f"Failed to store results in Redis: {e}")

        # Update Langfuse trace with summary
        if lf_trace:
            try:
                lf_trace.update(
                    output=summary,
                    metadata={
                        "run_id": run_id,
                        "models": models,
                        "prompt_sets": prompt_set_names,
                        "summary": summary,
                    },
                )
            except Exception:
                pass

        await broadcast({
            "type": "benchmark_complete",
            "run_id": run_id,
            "summary": summary,
        })

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        if lf_trace:
            try:
                lf_trace.update(level="ERROR", status_message=str(e))
            except Exception:
                pass
        await broadcast({
            "type": "benchmark_error",
            "run_id": run_id,
            "error": str(e),
        })
    finally:
        _active_benchmark = None
        # Flush Langfuse to ensure all events are sent
        if _langfuse:
            try:
                _langfuse.flush()
            except Exception:
                pass


async def _benchmark_single(
    model: str, prompt_set: str, prompt_meta: dict, run_id: str,
    backend: str = "ollama",
) -> BenchmarkResult:
    """Benchmark a single model+prompt pair with streaming token measurement."""
    result = BenchmarkResult(model, prompt_set, prompt_meta)

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0
            thinking_count = 0
            response_chunks = []
            thinking_chunks = []

            if backend == "vllm":
                # ── vLLM: use /v1/completions to bypass harmony parser crashes ─
                try:
                    mr = await client.get(f"{VLLM_BASE_URL}/v1/models")
                    vllm_model_id = mr.json()["data"][0]["id"]
                except Exception:
                    vllm_model_id = model

                # Format prompt using the model's chat template
                # gpt-oss uses: <|start|>system<|message|>...<|end|><|start|>user<|message|>...<|end|><|start|>assistant
                # Other models use standard ChatML: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
                if "gpt-oss" in vllm_model_id:
                    import datetime
                    today = datetime.date.today().strftime("%Y-%m-%d")
                    formatted_prompt = (
                        f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                        f"Knowledge cutoff: 2024-06\nCurrent date: {today}\n\n"
                        f"Reasoning: medium\n\n"
                        f"# Valid channels: analysis, commentary, final. Channel must be included for every message."
                        f"<|end|><|start|>user<|message|>{prompt_meta['prompt']}<|end|><|start|>assistant"
                    )
                else:
                    # Generic ChatML format for Qwen, Llama, Gemma etc.
                    formatted_prompt = (
                        f"<|im_start|>user\n{prompt_meta['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                    )

                # Use streaming /v1/completions (bypasses harmony parser, gives real TTFT)
                in_thinking = True  # gpt-oss starts with analysis/thinking channel
                async with client.stream(
                    "POST",
                    f"{VLLM_BASE_URL}/v1/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": vllm_model_id,
                        "prompt": formatted_prompt,
                        "max_tokens": 2048,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        text = choices[0].get("text", "")
                        if not text:
                            if choices[0].get("finish_reason"):
                                break
                            continue

                        token_count += 1
                        now = time.perf_counter()

                        if first_token_time is None:
                            first_token_time = now
                            result.ttft_ms = (first_token_time - start_time) * 1000

                        # Track gpt-oss channel transitions
                        # Special tokens come as bare words: "analysis", "final", "commentary"
                        # with empty-string tokens as separators
                        stripped = text.strip()
                        if stripped in ("analysis", "commentary"):
                            in_thinking = True
                        elif stripped == "final":
                            in_thinking = False
                        elif stripped in ("", "assistant"):
                            pass  # Skip separator/role tokens
                        elif in_thinking:
                            thinking_chunks.append(text)
                            thinking_count += 1
                        else:
                            response_chunks.append(text)

                        if token_count % 5 == 0:
                            elapsed = now - start_time
                            tps = token_count / elapsed if elapsed > 0 else 0
                            await broadcast({
                                "type": "token",
                                "run_id": run_id,
                                "model": model,
                                "prompt_id": prompt_meta["id"],
                                "token_num": token_count,
                                "tokens_per_sec": round(tps, 2),
                            })

                        if choices[0].get("finish_reason"):
                            break

            elif backend == "litellm":
                # ── LiteLLM: OpenAI-compatible streaming ──────────
                async with client.stream(
                    "POST",
                    f"{LITELLM_BASE_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_meta["prompt"]}],
                        "max_tokens": 2048,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        resp_text = delta.get("content", "")
                        think_text = delta.get("reasoning_content", "")
                        has_content = bool(resp_text) or bool(think_text)

                        if has_content:
                            token_count += 1
                            now = time.perf_counter()

                            if first_token_time is None:
                                first_token_time = now
                                result.ttft_ms = (first_token_time - start_time) * 1000

                            if resp_text:
                                response_chunks.append(resp_text)
                            if think_text:
                                thinking_chunks.append(think_text)
                                thinking_count += 1

                            if token_count % 5 == 0:
                                elapsed = now - start_time
                                tps = token_count / elapsed if elapsed > 0 else 0
                                await broadcast({
                                    "type": "token",
                                    "run_id": run_id,
                                    "model": model,
                                    "prompt_id": prompt_meta["id"],
                                    "token_num": token_count,
                                    "tokens_per_sec": round(tps, 2),
                                })

                        if choices[0].get("finish_reason"):
                            break
            else:
                # ── Ollama: native streaming ──────────────────────
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt_meta["prompt"],
                        "stream": True,
                        "options": {"num_predict": 2048},
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        resp_text = chunk.get("response", "")
                        think_text = chunk.get("thinking", "")
                        has_content = bool(resp_text) or bool(think_text)

                        if has_content:
                            token_count += 1
                            now = time.perf_counter()

                            if first_token_time is None:
                                first_token_time = now
                                result.ttft_ms = (first_token_time - start_time) * 1000

                            if resp_text:
                                response_chunks.append(resp_text)
                            if think_text:
                                thinking_chunks.append(think_text)
                                thinking_count += 1

                            if token_count % 5 == 0:
                                elapsed = now - start_time
                                tps = token_count / elapsed if elapsed > 0 else 0
                                await broadcast({
                                    "type": "token",
                                    "run_id": run_id,
                                    "model": model,
                                    "prompt_id": prompt_meta["id"],
                                    "token_num": token_count,
                                    "tokens_per_sec": round(tps, 2),
                                })

                        if chunk.get("done"):
                            break

            end_time = time.perf_counter()
            total_time = end_time - start_time

            result.total_tokens = token_count
            result.thinking_tokens = thinking_count
            result.total_time_ms = total_time * 1000
            result.tokens_per_sec = token_count / total_time if total_time > 0 else 0
            result.response_text = "".join(response_chunks)
            result.thinking_text = "".join(thinking_chunks)
            result.scores = score_response(prompt_meta, result.response_text)

    except Exception as e:
        result.error = str(e)
        logger.error(f"Benchmark error for {model}/{prompt_meta['id']}: {e}")

    return result


def _compute_summary(run_id: str, results: list[dict]) -> dict:
    """Compute aggregate summary statistics from benchmark results."""
    if not results:
        return {"run_id": run_id, "models": {}}

    model_stats = {}
    for r in results:
        if r.get("error"):
            continue
        model = r["model"]
        if model not in model_stats:
            model_stats[model] = {
                "tokens_per_sec": [],
                "ttft_ms": [],
                "total_tokens": [],
                "total_time_ms": [],
                "thinking_tokens": [],
                "categories": {},
                "correct": 0,
                "has_answer": 0,
            }
        stats = model_stats[model]
        stats["tokens_per_sec"].append(r["tokens_per_sec"])
        stats["ttft_ms"].append(r["ttft_ms"])
        stats["total_tokens"].append(r["total_tokens"])
        stats["total_time_ms"].append(r["total_time_ms"])
        stats["thinking_tokens"].append(r.get("thinking_tokens", 0))

        # Track accuracy
        scores = r.get("scores") or {}
        if "correct" in scores:
            stats["has_answer"] += 1
            if scores["correct"]:
                stats["correct"] += 1

        # Per-category tracking
        cat = r.get("category", "general")
        if cat not in stats["categories"]:
            stats["categories"][cat] = {"tps": [], "ttft": [], "count": 0, "correct": 0, "has_answer": 0}
        cat_data = stats["categories"][cat]
        cat_data["tps"].append(r["tokens_per_sec"])
        cat_data["ttft"].append(r["ttft_ms"])
        cat_data["count"] += 1
        if "correct" in scores:
            cat_data["has_answer"] += 1
            if scores["correct"]:
                cat_data["correct"] += 1

    summary = {"run_id": run_id, "models": {}, "timestamp": time.time()}
    for model, stats in model_stats.items():
        n = len(stats["tokens_per_sec"])
        cat_summary = {}
        for cat, cdata in stats["categories"].items():
            cn = cdata["count"]
            cat_summary[cat] = {
                "avg_tps": round(sum(cdata["tps"]) / cn, 2) if cn else 0,
                "avg_ttft": round(sum(cdata["ttft"]) / cn, 1) if cn else 0,
                "count": cn,
                "accuracy": round(cdata["correct"] / cdata["has_answer"] * 100, 1) if cdata["has_answer"] else None,
            }

        summary["models"][model] = {
            "prompts_completed": n,
            "avg_tokens_per_sec": round(sum(stats["tokens_per_sec"]) / n, 2) if n else 0,
            "avg_ttft_ms": round(sum(stats["ttft_ms"]) / n, 1) if n else 0,
            "avg_total_tokens": round(sum(stats["total_tokens"]) / n) if n else 0,
            "avg_total_time_ms": round(sum(stats["total_time_ms"]) / n, 1) if n else 0,
            "min_tokens_per_sec": round(min(stats["tokens_per_sec"]), 2) if n else 0,
            "max_tokens_per_sec": round(max(stats["tokens_per_sec"]), 2) if n else 0,
            "avg_thinking_tokens": round(sum(stats["thinking_tokens"]) / n) if n else 0,
            "accuracy": round(stats["correct"] / stats["has_answer"] * 100, 1) if stats["has_answer"] else None,
            "categories": cat_summary,
        }

    return summary


# ── History / Leaderboard / Run Detail ────────────────────────
@app.get("/api/history")
async def api_history():
    """Get benchmark history from Redis."""
    if not _redis_client:
        return {"results": []}

    try:
        keys = _redis_client.keys("specter:result:*")
        results = []
        for key in keys:
            data = _redis_client.get(key)
            if data:
                parsed = json.loads(data)
                results.append(parsed.get("summary", {}))

        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return {"results": results}
    except Exception as e:
        logger.warning(f"Failed to read history: {e}")
        return {"results": []}


@app.get("/api/run/{run_id}")
async def api_run_detail(run_id: str):
    """Get full run results including individual prompt results."""
    if not _redis_client:
        return JSONResponse({"error": "No persistence available"}, status_code=503)
    try:
        data = _redis_client.get(f"specter:result:{run_id}")
        if not data:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        return json.loads(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/run/{run_id}")
async def api_delete_run(run_id: str):
    """Delete a benchmark run from history."""
    if not _redis_client:
        return JSONResponse({"error": "No persistence available"}, status_code=503)
    try:
        deleted = _redis_client.delete(f"specter:result:{run_id}")
        if deleted:
            return {"status": "deleted", "run_id": run_id}
        return JSONResponse({"error": "Run not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/leaderboard")
async def api_leaderboard():
    """Get leaderboard: latest run per model across all runs."""
    if not _redis_client:
        return {"leaderboard": []}

    try:
        keys = _redis_client.keys("specter:result:*")
        latest_by_model = {}

        for key in keys:
            data = _redis_client.get(key)
            if not data:
                continue
            parsed = json.loads(data)
            summary = parsed.get("summary", {})
            # Skip incomplete/cancelled runs — don't overwrite finished results
            if not summary.get("completed", True):
                continue
            ts = summary.get("timestamp", 0)
            run_backend = summary.get("backend", "ollama")
            for model, stats in summary.get("models", {}).items():
                # Key by model+backend so same model with different backends gets separate entries
                lb_key = f"{model}|{run_backend}"
                if lb_key not in latest_by_model or ts > latest_by_model[lb_key]["_ts"]:
                    # Detect which engine served the model
                    if run_backend == "litellm":
                        # Infer from model name: vllm/* = vllm, otherwise ollama
                        engine = "vllm" if model.startswith("vllm/") else "ollama"
                    else:
                        engine = run_backend  # direct = engine is the backend itself

                    latest_by_model[lb_key] = {
                        "model": model,
                        "backend": run_backend,
                        "engine": engine,
                        "avg_tokens_per_sec": stats.get("avg_tokens_per_sec", 0),
                        "avg_ttft_ms": stats.get("avg_ttft_ms", 0),
                        "prompts_completed": stats.get("prompts_completed", 0),
                        "avg_total_tokens": stats.get("avg_total_tokens", 0),
                        "accuracy": stats.get("accuracy"),
                        "categories": stats.get("categories", {}),
                        "run_id": summary.get("run_id", ""),
                        "_ts": ts,
                    }

        # Strip internal timestamp and sort
        leaderboard = []
        for entry in latest_by_model.values():
            clean = {k: v for k, v in entry.items() if k != "_ts"}
            leaderboard.append(clean)
        leaderboard.sort(key=lambda x: x["avg_tokens_per_sec"], reverse=True)

        return {"leaderboard": leaderboard}
    except Exception as e:
        logger.warning(f"Failed to compute leaderboard: {e}")
        return {"leaderboard": []}


# ── WebSocket ──────────────────────────────────────────────────
@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    logger.info(f"WS connected ({len(_ws_clients)} total)")

    # Send current state
    if _active_benchmark:
        await ws.send_text(json.dumps({
            "type": "status",
            "benchmark_running": True,
            "run_id": _active_benchmark["run_id"],
        }))
    else:
        await ws.send_text(json.dumps({"type": "status", "benchmark_running": False}))

    # Send GPU stats immediately
    stats = await _get_gpu_stats()
    if stats:
        await ws.send_text(json.dumps({"type": "gpu", **stats}))

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd")
            if cmd == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS error: {e}")
    finally:
        _ws_clients.discard(ws)
        logger.info(f"WS disconnected ({len(_ws_clients)} total)")


# ── Static Files (MUST be last) ────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
