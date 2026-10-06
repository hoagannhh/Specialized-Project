"""
Tầng 3 — Suy luận (tự viết bằng PyTorch thuần)
==================================================

LƯU Ý QUAN TRỌNG (đã thống nhất): lúc suy luận KHÔNG còn khái niệm
Anchor/Positive/Negative — chỉ cần 2 câu (A, B) bất kỳ, đưa qua CÙNG
MỘT model BiEncoder đã train, rồi tính cosine similarity (ở đây là
dot product, vì embedding đã chuẩn hóa L2 sẵn trong model.py).
"""

import os

try:
    import torch
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers")

from model import BiEncoder, DEFAULT_PRETRAINED_NAME


AMBIGUOUS_LOW = 0.4
AMBIGUOUS_HIGH = 0.85

_cached_model = None
_cached_tokenizer = None


def load_cached_model(checkpoint_dir: str, pretrained_name: str = DEFAULT_PRETRAINED_NAME, use_cache: bool = True):
    """Load model + tokenizer đã train, có cache tránh load lại nhiều lần."""
    global _cached_model, _cached_tokenizer

    if use_cache and _cached_model is not None:
        return _cached_model, _cached_tokenizer

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiEncoder(pretrained_name=pretrained_name)
    checkpoint_path = os.path.join(checkpoint_dir, "tang3_best.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)  # tokenizer đã lưu cùng checkpoint lúc train

    if use_cache:
        _cached_model, _cached_tokenizer = model, tokenizer

    return model, tokenizer


@torch.no_grad()
def _encode(sentence: str, model, tokenizer, max_length: int = 128) -> "torch.Tensor":
    device = next(model.parameters()).device
    encoding = tokenizer(
        sentence, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    return model(input_ids, attention_mask)


def tang3_predict(
    sentence_a: str,
    sentence_b: str,
    checkpoint_dir: str = "checkpoints/tang3",
    model=None,
    tokenizer=None,
) -> dict:
    """
    Hàm CHÍNH của Tầng 3.

    Output decision:
        "ambiguous" : rơi vào vùng mập mờ -> cần Tầng 4 tinh chỉnh
        "confident" : similarity đủ rõ ràng -> có thể dùng kết quả
                      này luôn, bỏ qua Tầng 4
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_cached_model(checkpoint_dir)

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"similarity": 0.0, "decision": "confident", "needs_tang4": False}

    emb_a = _encode(str(sentence_a), model, tokenizer)
    emb_b = _encode(str(sentence_b), model, tokenizer)

    # Embedding đã chuẩn hóa L2 -> dot product = cosine similarity
    similarity = float(torch.sum(emb_a * emb_b, dim=1).item())

    is_ambiguous = AMBIGUOUS_LOW <= similarity <= AMBIGUOUS_HIGH

    return {
        "similarity": similarity,
        "decision": "ambiguous" if is_ambiguous else "confident",
        "needs_tang4": is_ambiguous,
    }


def tang3_predict_batch(pairs: list, checkpoint_dir: str = "checkpoints/tang3"):
    """Phiên bản hàng loạt — encode theo batch, nhanh hơn gọi từng cặp."""
    import pandas as pd

    model, tokenizer = load_cached_model(checkpoint_dir)
    device = next(model.parameters()).device

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    with torch.no_grad():
        enc_a = tokenizer(sentences_a, truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc_b = tokenizer(sentences_b, truncation=True, max_length=128, padding=True, return_tensors="pt")

        emb_a = model(enc_a["input_ids"].to(device), enc_a["attention_mask"].to(device))
        emb_b = model(enc_b["input_ids"].to(device), enc_b["attention_mask"].to(device))

        similarities = torch.sum(emb_a * emb_b, dim=1).cpu().numpy()

    is_ambiguous = (similarities >= AMBIGUOUS_LOW) & (similarities <= AMBIGUOUS_HIGH)

    return pd.DataFrame({
        "sentence_a": sentences_a, "sentence_b": sentences_b,
        "similarity": similarities, "needs_tang4": is_ambiguous,
    })


if __name__ == "__main__":
    result = tang3_predict(
        "What is the story of the Kohinoor Diamond?",
        "Can you tell me about the history of the Kohinoor Diamond?",
        checkpoint_dir="checkpoints/tang3",
    )
    print(result)
