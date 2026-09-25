import hashlib
import json
from typing import Any

from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from .config import settings
from .core import embed
from .metrics import CACHE_HITS, CACHE_SIZE

class Cache:
    def __init__(self):
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.ready = False

    async def init(self):
        try:
            collections = await self.qdrant.get_collections()
            if settings.qdrant_collection not in [c.name for c in collections.collections]:
                await self.qdrant.create_collection(settings.qdrant_collection, vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE))
            self.ready = True
        except Exception:
            self.ready = False

    def exact_key(self, task: str, text: str, prompt_version: str, model_version: str) -> str:
        raw = f"{task}|{prompt_version}|{model_version}|{text}"
        return "ai-gateway:exact:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get_exact(self, key: str):
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def put_exact(self, key: str, value: Any, ttl: int = 86400):
        await self.redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)

    async def search_semantic(self, task: str, text: str, threshold: float):
        if not self.ready:
            return None
        try:
            result = await self.qdrant.search(settings.qdrant_collection, query_vector=embed(text), limit=1, query_filter=models.Filter(must=[models.FieldCondition(key="task_type", match=models.MatchValue(value=task))]))
            if result and result[0].score >= threshold:
                payload = result[0].payload or {}
                CACHE_HITS.labels("semantic", task).inc()
                return {"score": result[0].score, **payload}
        except Exception:
            return None
        return None

    async def put_semantic(self, task: str, text: str, value: dict):
        if not self.ready:
            return
        try:
            point_id = hashlib.sha256(f"{task}|{text}".encode()).hexdigest()[:32]
            await self.qdrant.upsert(settings.qdrant_collection, points=[models.PointStruct(id=point_id, vector=embed(text), payload={"task_type": task, "text": text, "result": value})])
        except Exception:
            return

cache = Cache()
