from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter("ai_gateway_requests_total", "Total gateway requests", ["task_type", "status"])
CACHE_HITS = Counter("ai_gateway_cache_hits_total", "Cache hits", ["cache_type", "task_type"])
RAW_TOKENS = Counter("ai_gateway_raw_input_tokens_total", "Raw input tokens", ["task_type"])
SENT_TOKENS = Counter("ai_gateway_sent_input_tokens_total", "Sent input tokens", ["task_type"])
SAVED_TOKENS = Counter("ai_gateway_saved_tokens_total", "Saved input tokens", ["task_type"])
COMPRESSION_SECONDS = Histogram("ai_gateway_compression_seconds", "Compression time", ["task_type"])
PROVIDER_SECONDS = Histogram("ai_gateway_provider_seconds", "Provider latency", ["task_type"])
CACHE_SIZE = Gauge("ai_gateway_semantic_cache_size", "Semantic cache size")
