"""
Tầng 4 — Suy luận (Cross-Encoder đã huấn luyện)
==================================================

CrossEncoder.predict() tự động ghép [CLS] A [SEP] B [SEP] bên trong,
người dùng chỉ cần truyền 2 câu riêng biệt, không cần tự ghép tay.
"""

from model import load_finetuned_cross_encoder


_cached_model = None


def load_cross_encoder(checkpoint_dir: str, use_cache: bool = True):
    """
    Wrapper quanh model.load_finetuned_cross_encoder(), có thêm cache
    cấp module — cùng nguyên tắc với Tầng 3 (model.py lo khởi tạo
    đúng kiến trúc, file này lo tái sử dụng hiệu quả).
    """
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model

    model = load_finetuned_cross_encoder(checkpoint_dir)

    if use_cache:
        _cached_model = model

    return model


def tang4_predict(
    sentence_a: str,
    sentence_b: str,
    checkpoint_dir: str = "checkpoints/tang4",
    model=None,
    threshold: float = 0.5,
) -> dict:
    """
    Hàm CHÍNH của Tầng 4 — bước tinh chỉnh cuối cùng trước khi sang
    Tầng 5 (giải thích).

    CrossEncoder.predict() với num_labels=1 trả về giá trị đã qua
    sigmoid (0-1) nếu model được tạo với apply_softmax phù hợp —
    đây chính là "% xác suất đạo văn" THẬT SỰ (khác similarity score
    của Tầng 1-3, như đã phân biệt rõ ở phần thiết kế output).
    """
    if model is None:
        model = load_cross_encoder(checkpoint_dir)

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"probability": 0.0, "label": "không đạo văn"}

    probability = float(model.predict([(str(sentence_a), str(sentence_b))])[0])

    label = "đạo văn (AI-paraphrase)" if probability >= threshold else "không đạo văn"

    return {"probability": probability, "label": label}


def tang4_predict_batch(
    pairs: list,
    checkpoint_dir: str = "checkpoints/tang4",
    model=None,
    threshold: float = 0.5,
):
    import pandas as pd

    if model is None:
        model = load_cross_encoder(checkpoint_dir)

    str_pairs = [(str(a), str(b)) for a, b in pairs]
    probabilities = model.predict(str_pairs)

    labels = ["đạo văn (AI-paraphrase)" if p >= threshold else "không đạo văn" for p in probabilities]

    return pd.DataFrame({
        "sentence_a": [p[0] for p in str_pairs],
        "sentence_b": [p[1] for p in str_pairs],
        "probability": probabilities,
        "label": labels,
    })


if __name__ == "__main__":
    result = tang4_predict(
        "How do I use Google's image search?",
        "What is the method for conducting a search using Google's image search?",
        checkpoint_dir="checkpoints/tang4",
    )
    print(result)
