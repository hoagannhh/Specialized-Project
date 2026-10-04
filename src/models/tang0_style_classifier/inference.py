"""
Tầng 0 — Suy luận (dùng model + ngưỡng đã train)
===================================================

Load checkpoint đã train (model.py + train.py), có cache trong bộ
nhớ để dùng lặp lại cho nhiều câu mà không phải load lại mỗi lần
(cùng nguyên tắc với Tầng 1/2).
"""

import os

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers")

from model import SingleSentenceClassifier, DEFAULT_PRETRAINED_NAME


_cached_model = None
_cached_tokenizer = None
_cached_threshold = None


def load_for_inference(
    checkpoint_dir: str,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    use_cache: bool = True,
):
    """
    Load model + tokenizer + ngưỡng đã lưu từ train.py.
    """
    global _cached_model, _cached_tokenizer, _cached_threshold

    if use_cache and _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SingleSentenceClassifier(pretrained_name=pretrained_name)
    checkpoint_path = os.path.join(checkpoint_dir, "tang0_best.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    threshold_path = os.path.join(checkpoint_dir, "threshold.txt")
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    if use_cache:
        _cached_model, _cached_tokenizer, _cached_threshold = model, tokenizer, threshold

    return model, tokenizer, threshold


def predict_sentence(
    sentence: str,
    checkpoint_dir: str = "checkpoints/tang0",
    max_length: int = 128,
) -> dict:
    """
    Hàm CHÍNH dùng trong pipeline hệ thống mở rộng (input 1 file).

    Output:
        {
            "probability": xác suất câu này là AI-touched (0-1),
            "decision": "suspicious" (vượt ngưỡng, cần Web Search)
                        hoặc "skip" (bỏ qua, không nghi ngờ)
        }
    """
    model, tokenizer, threshold = load_for_inference(checkpoint_dir)
    device = next(model.parameters()).device

    encoding = tokenizer(
        str(sentence),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    probs = model.predict_proba(input_ids, attention_mask)
    probability_ai = float(probs[0, 1].item())

    decision = "suspicious" if probability_ai >= threshold else "skip"

    return {"probability": probability_ai, "decision": decision, "threshold_used": threshold}


def predict_batch(
    sentences: list,
    checkpoint_dir: str = "checkpoints/tang0",
    max_length: int = 128,
    batch_size: int = 32,
):
    """
    Phiên bản hàng loạt — xử lý nhiều câu cùng lúc theo batch, nhanh
    hơn gọi predict_sentence() từng câu trong vòng lặp Python.
    """
    import pandas as pd

    model, tokenizer, threshold = load_for_inference(checkpoint_dir)
    device = next(model.parameters()).device

    all_probs = []
    for i in range(0, len(sentences), batch_size):
        batch = [str(s) for s in sentences[i:i + batch_size]]
        encoding = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        probs = model.predict_proba(input_ids, attention_mask)
        all_probs.extend(probs[:, 1].cpu().tolist())

    decisions = ["suspicious" if p >= threshold else "skip" for p in all_probs]

    return pd.DataFrame({
        "sentence": sentences,
        "probability": all_probs,
        "decision": decisions,
    })


if __name__ == "__main__":
    result = predict_sentence("The rapid development of artificial intelligence has changed education.")
    print(result)
