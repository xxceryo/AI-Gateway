from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .cache import cache
from .config import settings
from .core import compress_text, count_tokens, normalize, provider_call
from .metrics import CACHE_HITS, RAW_TOKENS, REQUESTS, SAVED_TOKENS, SENT_TOKENS

class ProcessRequest(BaseModel):
    task_type: str = Field(default="customer_potential")
    text: str = Field(min_length=1)
    user_id: str | None = None
    compression_level: str = Field(default="aggressive", pattern="^(safe|balanced|aggressive)$")
    max_input_tokens: int | None = Field(default=None, ge=80, le=100000)
    prompt_version: str = "v1"
    model_version: str = "provider-default"

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.init()
    yield
    await cache.redis.aclose()
    await cache.qdrant.close()

app = FastAPI(title="AI-Gateway", version="0.1.0", lifespan=lifespan)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "qdrant_ready": cache.ready}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/process")
async def process(req: ProcessRequest):
    task = req.task_type
    raw = normalize(req.text)
    raw_tokens = count_tokens(raw)
    RAW_TOKENS.labels(task).inc(raw_tokens)
    max_tokens = req.max_input_tokens or settings.default_max_input_tokens
    model_version = f"{settings.upstream_base_url}|{settings.upstream_model}|{req.model_version}"
    exact_key = cache.exact_key(task, raw, req.prompt_version, model_version)
    exact = await cache.get_exact(exact_key)
    if exact:
        CACHE_HITS.labels("exact", task).inc()
        REQUESTS.labels(task, "exact_cache").inc()
        return {**exact, "usage": {**exact["usage"], "cache_hit": True, "cache_type": "exact"}}

    compressed, kept, removed, sent_tokens = compress_text(raw, task, max_tokens, req.compression_level)
    SENT_TOKENS.labels(task).inc(sent_tokens)
    SAVED_TOKENS.labels(task).inc(max(0, raw_tokens - sent_tokens))
    direct_threshold = settings.semantic_direct_threshold
    semantic_namespace = f"{task}|{model_version}|{req.prompt_version}"
    semantic = await cache.search_semantic(semantic_namespace, compressed, direct_threshold) if task == "customer_potential" and settings.upstream_base_url else None
    if semantic:
        result = semantic.get("result", {})
        response = {"result": result, "compressed_text": compressed, "evidence": {"kept": [c.chunk_id for c in kept], "removed": [c.chunk_id for c in removed]}, "usage": {"raw_input_tokens": raw_tokens, "sent_input_tokens": sent_tokens, "saved_input_tokens": max(0, raw_tokens - sent_tokens), "cache_hit": True, "cache_type": "semantic_direct"}}
        await cache.put_exact(exact_key, response)
        REQUESTS.labels(task, "semantic_cache").inc()
        return response

    provider = await provider_call(task, compressed)
    result = provider.get("choices", [{}])[0].get("message", {}).get("content", provider)
    provider_usage = provider.get("usage", {})
    response = {"result": result, "compressed_text": compressed, "evidence": {"kept": [c.chunk_id for c in kept], "removed": [c.chunk_id for c in removed]}, "usage": {"raw_input_tokens": raw_tokens, "sent_input_tokens": sent_tokens, "saved_input_tokens": max(0, raw_tokens - sent_tokens), "provider_prompt_tokens": provider_usage.get("prompt_tokens"), "provider_completion_tokens": provider_usage.get("completion_tokens"), "provider_cache_hit_tokens": provider_usage.get("prompt_cache_hit_tokens"), "cache_hit": False, "cache_type": None}}
    if settings.upstream_base_url:
        await cache.put_exact(exact_key, response)
        if task == "customer_potential":
            await cache.put_semantic(semantic_namespace, compressed, result)
    REQUESTS.labels(task, "provider").inc()
    return response
