# AI-Gateway

Open-source-first text-to-LLM gateway for customer-potential mining and KYC.

AI-Gateway accepts unstructured text, performs token-aware extractive compression, exact/semantic caching, dynamic budgets, provider forwarding, and observability. It does not require upstream data reformatting or fine-tuning the provider model.

## Quick start

```bash
cp .env.example .env
cd middleware && docker compose up -d
cd ..
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Open:

- API docs: http://localhost:8080/docs
- Metrics: http://localhost:8080/metrics
- Grafana: http://localhost:3000 (admin/admin)
- Qdrant: http://localhost:6333/dashboard

The gateway works in dry-run mode without an upstream API key. Set `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, and `UPSTREAM_MODEL` to forward cache misses to an OpenAI-compatible provider.

## API

`POST /v1/process`

```json
{
  "task_type": "customer_potential",
  "text": "Customer says the price is high but wants to launch this month.",
  "user_id": "u-123",
  "compression_level": "aggressive",
  "max_input_tokens": 600
}
```

## Design

- Exact cache: Redis/Valkey.
- Semantic cache: Qdrant, with deterministic local hashing embeddings by default; plug in a provider embedding endpoint with `EMBEDDING_BASE_URL`.
- Compression: sentence extraction, token budget, task-specific keyword protection, near-duplicate removal.
- Observability: Prometheus + Grafana. Every request records raw/sent tokens, cache layer, compression latency, provider latency, and selected chunks.

## Development

```bash
pytest -q
```

This project intentionally separates raw evidence from the compressed prompt. The API response includes source chunk IDs so downstream systems can audit what was kept and removed.
