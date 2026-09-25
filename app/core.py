import hashlib
import math
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import httpx
import tiktoken

from .config import settings
from .metrics import COMPRESSION_SECONDS, PROVIDER_SECONDS

_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;.!?])\s+|\n+")
_TASK_TERMS = {
    "customer_potential": ["购买", "预算", "报价", "价格", "上线", "需求", "供应商", "合同", "采购", "试用", "客户", "购买意向"],
    "kyc": ["身份", "姓名", "公司", "机构", "金额", "交易", "风险", "异常", "关系", "职业", "地址", "证件", "来源", "受益人"],
}
_PROTECTED = re.compile(r"(金额|人民币|美元|元|%|百分之|身份证|护照|合同|账号|账户|风险|异常|否认|但是|然而|尚未|无法确认|除非|可能|不|没有)")

@dataclass
class Chunk:
    chunk_id: str
    text: str
    index: int
    tokens: int
    score: float = 0.0


def tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = tokenizer()
    return len(enc.encode(text)) if enc else max(1, math.ceil(len(text) / 4))


def normalize(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def split_chunks(text: str) -> list[Chunk]:
    parts = [p.strip() for p in _SENTENCE_RE.split(normalize(text)) if p.strip()]
    result = []
    for i, part in enumerate(parts):
        cid = hashlib.sha256(f"{i}:{part}".encode()).hexdigest()[:16]
        result.append(Chunk(cid, part, i, count_tokens(part)))
    return result


def _hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    # Deterministic fallback; production can use a provider embedding endpoint.
    vec = [0.0] * dimensions
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        vec[h % dimensions] += 1.0 if h % 2 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x*x for x in a)) or 1) * (math.sqrt(sum(y*y for y in b)) or 1))


def embed(text: str) -> list[float]:
    return _hash_embedding(text)


def compress_text(text: str, task_type: str, max_tokens: int, level: str) -> tuple[str, list[Chunk], list[Chunk], float]:
    started = perf_counter()
    chunks = split_chunks(text)
    terms = _TASK_TERMS.get(task_type, [])
    deduped: list[Chunk] = []
    seen = set()
    for c in chunks:
        fingerprint = hashlib.sha256(re.sub(r"\W", "", c.text.lower()).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        score = sum(2.0 for t in terms if t in c.text)
        score += 2.0 if _PROTECTED.search(c.text) else 0.0
        score += 1.0 / (1 + c.index * 0.03)
        c.score = score
        deduped.append(c)
    if level == "safe":
        keep_budget = max_tokens
    elif level == "aggressive":
        keep_budget = max(80, int(max_tokens * 0.8))
    else:
        keep_budget = int(max_tokens * 0.95)
    selected, used = [], 0
    # Preserve source order after ranking, while allowing the most relevant chunks first.
    for c in sorted(deduped, key=lambda x: (x.score, -x.index), reverse=True):
        if used + c.tokens <= keep_budget or not selected:
            selected.append(c)
            used += c.tokens
    selected.sort(key=lambda x: x.index)
    selected_ids = {c.chunk_id for c in selected}
    removed = [c for c in chunks if c.chunk_id not in selected_ids]
    output = "\n".join(c.text for c in selected)
    COMPRESSION_SECONDS.labels(task_type).observe(perf_counter() - started)
    return output, selected, removed, count_tokens(output)


async def provider_call(task_type: str, text: str) -> dict:
    if not settings.upstream_base_url:
        return {"mode": "dry_run", "task_type": task_type, "decision": "provider_not_configured", "input_preview": text[:300]}
    headers = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    payload = {"model": settings.upstream_model, "messages": [{"role": "user", "content": text}], "temperature": 0}
    started = perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(settings.upstream_base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    PROVIDER_SECONDS.labels(task_type).observe(perf_counter() - started)
    return data
