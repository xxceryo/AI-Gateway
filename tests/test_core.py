from app.core import compress_text, count_tokens, split_chunks

def test_compression_keeps_task_relevant_text():
    raw = "你好，最近还好吗？ 客户表示预算有限，但希望本月完成上线。 客户询问报价。"
    compressed, kept, removed, sent = compress_text(raw, "customer_potential", 100, "aggressive")
    assert "本月完成上线" in compressed
    assert sent <= count_tokens(raw)
    assert len(kept) >= 1

def test_empty_not_allowed_by_api_model():
    assert count_tokens("hello") > 0

def test_chinese_punctuation_splits_without_spaces():
    assert len(split_chunks("客户询问报价。客户希望本月上线。")) == 2

def test_long_chunk_respects_hard_budget():
    compressed, *_ = compress_text("客户希望本月上线。" * 300, "customer_potential", 80, "aggressive")
    assert count_tokens(compressed) <= 80
