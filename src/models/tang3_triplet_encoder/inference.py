"""
Tầng 3 — Suy luận (dùng model đã fine-tune bằng Triplet Loss)
================================================================

LƯU Ý QUAN TRỌNG (đã thống nhất): lúc suy luận KHÔNG còn khái niệm
Anchor/Positive/Negative nữa — chỉ cần 2 câu (A, B) bất kỳ, đưa qua
CÙNG MỘT encoder đã train (vì train dùng 1 encoder chia sẻ trọng số
cho cả 3 nhánh), rồi tính cosine similarity giữa 2 embedding.
"""

try:
    from sentence_transformers.util import cos_sim
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install sentence-transformers")

from model import load_finetuned_encoder


# Ngưỡng "vùng mập mờ" theo sơ đồ: Tầng 3 luôn trả về similarity
# chính xác hơn Tầng 2, nhưng vẫn có thể cần Tầng 4 tinh chỉnh nếu
# rơi vào khoảng mập mờ này
AMBIGUOUS_LOW = 0.4
AMBIGUOUS_HIGH = 0.85


_cached_model = None


def load_cached_model(checkpoint_dir: str, use_cache: bool = True):
    """
    Wrapper quanh model.load_finetuned_encoder(), có thêm cache cấp
    module để không load lại model mỗi lần gọi (model.py chỉ lo việc
    KHỞI TẠO đúng kiến trúc, file này lo việc TÁI SỬ DỤNG hiệu quả).
    """
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model

    model = load_finetuned_encoder(checkpoint_dir)

    if use_cache:
        _cached_model = model

    return model


def tang3_predict(
    sentence_a: str,
    sentence_b: str,
    checkpoint_dir: str = "checkpoints/tang3",
    model=None,
) -> dict:
    """
    Hàm CHÍNH của Tầng 3.

    Output decision:
        "ambiguous"  : rơi vào vùng mập mờ -> cần Tầng 4 tinh chỉnh
        "confident"  : similarity đủ rõ ràng (cao hoặc thấp) -> có
                       thể dùng kết quả này luôn, bỏ qua Tầng 4
    """
    if model is None:
        model = load_cached_model(checkpoint_dir)

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"similarity": 0.0, "decision": "confident", "needs_tang4": False}

    embeddings = model.encode([str(sentence_a), str(sentence_b)], normalize_embeddings=True)
    similarity = float(cos_sim(embeddings[0], embeddings[1]).item())

    is_ambiguous = AMBIGUOUS_LOW <= similarity <= AMBIGUOUS_HIGH

    return {
        "similarity": similarity,
        "decision": "ambiguous" if is_ambiguous else "confident",
        "needs_tang4": is_ambiguous,
    }


def tang3_predict_batch(pairs: list, checkpoint_dir: str = "checkpoints/tang3", model=None):
    """Phiên bản hàng loạt, encode tất cả câu 1 lần."""
    import numpy as np
    import pandas as pd

    if model is None:
        model = load_cached_model(checkpoint_dir)

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    emb_a = model.encode(sentences_a, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(sentences_b, normalize_embeddings=True, show_progress_bar=False)

    similarities = np.sum(emb_a * emb_b, axis=1)
    is_ambiguous = (similarities >= AMBIGUOUS_LOW) & (similarities <= AMBIGUOUS_HIGH)

    return pd.DataFrame({
        "sentence_a": sentences_a,
        "sentence_b": sentences_b,
        "similarity": similarities,
        "needs_tang4": is_ambiguous,
    })


if __name__ == "__main__":
    result = tang3_predict(
        "What is the story of the Kohinoor Diamond?",
        "Can you tell me about the history of the Kohinoor Diamond?",
        checkpoint_dir="checkpoints/tang3",
    )
    print(result)
