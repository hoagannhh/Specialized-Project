"""
Tầng 2 — So sánh ngữ nghĩa sơ bộ (tự viết bằng transformers thuần)
======================================================================

KHÔNG CẦN TRAIN — chỉ tải model PRETRAINED có sẵn từ HuggingFace và
dùng ngay. Dùng CÙNG kiến trúc Mean Pooling với Tầng 3 (BERT encoder
+ mean pooling + L2 normalize), nhưng với trọng số GỐC (chưa
fine-tune) — đây chính là điểm khác biệt duy nhất giữa Tầng 2 và
Tầng 3 về mặt kỹ thuật.

Việc 2 tầng dùng chung kiến trúc nhưng khác trọng số (1 bên pretrained
gốc, 1 bên fine-tuned) là lý do hợp lý để có 2 tầng riêng biệt trong
pipeline: Tầng 2 nhanh/rẻ (không cần GPU train), Tầng 3 chính xác
hơn cho đúng domain bài toán (đã học từ dữ liệu paraphrase cụ thể).
"""

import numpy as np
import torch  # BẮT BUỘC — nhất quán với Tầng 0/3/4 (không cố "gracefully
               # degrade" khi thiếu torch, vì toàn bộ logic encode đều cần nó)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install transformers")


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Dùng checkpoint PRETRAINED của cộng đồng (đã được huấn luyện sẵn
# cho bài toán sentence similarity) qua transformers.AutoModel —
# không cần cài thư viện sentence-transformers, chỉ cần đúng tên
# checkpoint trên HuggingFace Hub.

LOW_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.85


def _mean_pooling(token_embeddings, attention_mask):
    """Giống hệt logic trong tang3/model.py — xem giải thích chi tiết ở đó."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


_cached_model = None
_cached_tokenizer = None


def load_model(model_name: str = DEFAULT_MODEL_NAME, use_cache: bool = True):
    """Load model + tokenizer pretrained, có cache tránh load lại nhiều lần."""
    global _cached_model, _cached_tokenizer

    if use_cache and _cached_model is not None:
        return _cached_model, _cached_tokenizer

    print(f"Đang tải model '{model_name}' từ HuggingFace (lần đầu sẽ chậm)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_cache:
        _cached_model, _cached_tokenizer = model, tokenizer

    return model, tokenizer


@torch.no_grad()
def _encode_batch(sentences: list, model, tokenizer, max_length: int = 128):
    device = next(model.parameters()).device
    encoding = tokenizer(
        sentences, truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = _mean_pooling(outputs.last_hidden_state, attention_mask)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings


def sbert_filter(
    sentence_a: str,
    sentence_b: str,
    model=None,
    tokenizer=None,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> dict:
    """
    Hàm CHÍNH của Tầng 2.

    Output decision có 3 khả năng:
        "stop"           : similarity quá thấp, loại bỏ
        "early_positive"  : similarity quá cao, có thể kết luận sớm
        "continue"        : vùng mập mờ, cần Tầng 3 phân tích sâu hơn
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"similarity": 0.0, "decision": "stop", "reason": "một trong hai câu rỗng"}

    embeddings = _encode_batch([str(sentence_a), str(sentence_b)], model, tokenizer)
    similarity = float(torch.sum(embeddings[0] * embeddings[1]).item())

    if similarity < low_threshold:
        decision, reason = "stop", f"similarity < {low_threshold}"
    elif similarity > high_threshold:
        decision, reason = "early_positive", f"similarity > {high_threshold}"
    else:
        decision, reason = "continue", None

    return {"similarity": similarity, "decision": decision, "reason": reason}


def sbert_filter_batch(pairs: list, model=None, tokenizer=None,
                        low_threshold: float = LOW_THRESHOLD, high_threshold: float = HIGH_THRESHOLD):
    """Phiên bản hàng loạt — encode tất cả câu 1 lần, nhanh hơn nhiều."""
    import pandas as pd

    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    emb_a = _encode_batch(sentences_a, model, tokenizer)
    emb_b = _encode_batch(sentences_b, model, tokenizer)

    similarities = torch.sum(emb_a * emb_b, dim=1).cpu().numpy()

    decisions = np.where(
        similarities < low_threshold, "stop",
        np.where(similarities > high_threshold, "early_positive", "continue")
    )

    return pd.DataFrame({
        "sentence_a": sentences_a, "sentence_b": sentences_b,
        "similarity": similarities, "decision": decisions,
    })


if __name__ == "__main__":
    result = sbert_filter(
        "The economy grew rapidly last year.",
        "Last year, the economy expanded at a fast pace.",
    )
    print(result)
