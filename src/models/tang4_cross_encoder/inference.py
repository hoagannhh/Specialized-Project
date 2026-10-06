"""
Tầng 4 — Suy luận (tự viết bằng PyTorch thuần)
==================================================
"""

import os

try:
    import torch
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers")

from model import CrossEncoderModel, DEFAULT_PRETRAINED_NAME


_cached_model = None
_cached_tokenizer = None


def load_cached_model(checkpoint_dir: str, pretrained_name: str = DEFAULT_PRETRAINED_NAME, use_cache: bool = True):
    global _cached_model, _cached_tokenizer

    if use_cache and _cached_model is not None:
        return _cached_model, _cached_tokenizer

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossEncoderModel(pretrained_name=pretrained_name)
    checkpoint_path = os.path.join(checkpoint_dir, "tang4_best.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    if use_cache:
        _cached_model, _cached_tokenizer = model, tokenizer

    return model, tokenizer


def tang4_predict(
    sentence_a: str,
    sentence_b: str,
    checkpoint_dir: str = "checkpoints/tang4",
    threshold: float = 0.5,
) -> dict:
    """
    Hàm CHÍNH của Tầng 4 — bước tinh chỉnh cuối cùng trước Tầng 5.

    Trả về xác suất THẬT SỰ (đã qua sigmoid, được train trực tiếp
    bằng BCEWithLogitsLoss) — khác similarity score của Tầng 1-3.
    """
    model, tokenizer = load_cached_model(checkpoint_dir)
    device = next(model.parameters()).device

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"probability": 0.0, "label": "không đạo văn"}

    encoding = tokenizer(
        str(sentence_a), str(sentence_b),
        truncation=True, max_length=256, padding="max_length", return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    probability = float(model.predict_proba(input_ids, attention_mask, token_type_ids).item())

    label = "đạo văn (AI-paraphrase)" if probability >= threshold else "không đạo văn"

    return {"probability": probability, "label": label}


def tang4_predict_batch(pairs: list, checkpoint_dir: str = "checkpoints/tang4", threshold: float = 0.5):
    import pandas as pd

    model, tokenizer = load_cached_model(checkpoint_dir)
    device = next(model.parameters()).device

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    encoding = tokenizer(
        sentences_a, sentences_b,
        truncation=True, max_length=256, padding=True, return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    probabilities = model.predict_proba(input_ids, attention_mask, token_type_ids).cpu().numpy()
    labels = ["đạo văn (AI-paraphrase)" if p >= threshold else "không đạo văn" for p in probabilities]

    return pd.DataFrame({
        "sentence_a": sentences_a, "sentence_b": sentences_b,
        "probability": probabilities, "label": labels,
    })


if __name__ == "__main__":
    result = tang4_predict(
        "How do I use Google's image search?",
        "What is the method for conducting a search using Google's image search?",
        checkpoint_dir="checkpoints/tang4",
    )
    print(result)
