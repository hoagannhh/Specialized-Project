"""
Tầng 2 — So sánh ngữ nghĩa sơ bộ (Sentence-BERT pretrained)
=============================================================

VAI TRÒ: giống hệt Tầng 1 về cấu trúc input/output (nhận 2 câu cụ
thể, trả về similarity + quyết định dừng/tiếp tục), chỉ khác kỹ
thuật: dùng Sentence-BERT ĐÃ PRETRAINED (KHÔNG fine-tune), thay vì
TF-IDF — giúp bắt được tương đồng NGỮ NGHĨA thay vì chỉ từ vựng bề
mặt.

KHÔNG CẦN TRAIN — chỉ cần tải model có sẵn từ HuggingFace và dùng
ngay. Đây là lý do thư mục này không có train.py.

Quy tắc ngưỡng theo sơ đồ:
    - similarity > 0.85  -> kết luận sớm (continue mạnh, có thể báo
                            đạo văn ngay nếu muốn tối ưu tốc độ)
    - similarity < 0.2   -> dừng (loại bỏ, không liên quan)
    - còn lại            -> "vùng mập mờ", chuyển tiếp sang Tầng 3
"""

from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # cho phép file này được import/đọc code
    # ngay cả khi chưa cài sentence-transformers (ví dụ lúc review code)


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# Model đa ngôn ngữ (hỗ trợ cả tiếng Việt lẫn tiếng Anh), nhẹ (~470MB),
# phù hợp pretrained similarity mà không cần fine-tune thêm.
# Lựa chọn thay thế nhẹ hơn (chỉ tiếng Anh): "all-MiniLM-L6-v2"

LOW_THRESHOLD = 0.2    # Dưới ngưỡng này -> dừng, loại bỏ
HIGH_THRESHOLD = 0.85  # Trên ngưỡng này -> kết luận sớm


_cached_model = None  # cache cấp module, tránh load lại model nhiều lần


def load_model(model_name: str = DEFAULT_MODEL_NAME, use_cache: bool = True) -> "SentenceTransformer":
    """
    Load Sentence-BERT pretrained, có cache trong bộ nhớ — quan trọng
    vì model này tương đối nặng (vài trăm MB), không nên load lại cho
    mỗi câu khi lặp qua nhiều câu trong 1 file.
    """
    global _cached_model
    if use_cache and _cached_model is not None:
        return _cached_model

    if SentenceTransformer is None:
        raise ImportError(
            "Chưa cài thư viện sentence-transformers. Chạy: "
            "pip install sentence-transformers"
        )

    print(f"Đang tải model '{model_name}' từ HuggingFace (lần đầu sẽ chậm)...")
    model = SentenceTransformer(model_name)

    if use_cache:
        _cached_model = model

    return model


def sbert_filter(
    sentence_a: str,
    sentence_b: str,
    model=None,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> dict:
    """
    Hàm CHÍNH của Tầng 2.

    Output decision có 3 khả năng (khác Tầng 1 chỉ có 2):
        "stop"             : similarity quá thấp, loại bỏ
        "early_positive"    : similarity quá cao, có thể kết luận sớm
                              là đạo văn (tùy chiến lược, có thể vẫn
                              cho đi tiếp để Tầng 5 giải thích)
        "continue"          : vùng mập mờ, cần Tầng 3 phân tích sâu hơn
    """
    if model is None:
        model = load_model()

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"similarity": 0.0, "decision": "stop", "reason": "một trong hai câu rỗng"}

    embeddings = model.encode(
        [str(sentence_a), str(sentence_b)],
        normalize_embeddings=True,  # chuẩn hóa L2 sẵn -> dot product = cosine similarity
        show_progress_bar=False,
    )
    similarity = float(np.dot(embeddings[0], embeddings[1]))

    if similarity < low_threshold:
        decision = "stop"
        reason = f"similarity < {low_threshold}"
    elif similarity > high_threshold:
        decision = "early_positive"
        reason = f"similarity > {high_threshold}"
    else:
        decision = "continue"
        reason = None

    return {"similarity": similarity, "decision": decision, "reason": reason}


def sbert_filter_batch(
    pairs: list,
    model=None,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
):
    """
    Phiên bản hàng loạt — encode TẤT CẢ câu 1 lần (batch), nhanh hơn
    nhiều so với gọi sbert_filter() từng cặp trong vòng lặp, vì
    SentenceTransformer tối ưu rất tốt cho xử lý theo batch (tận dụng
    GPU nếu có).
    """
    import pandas as pd

    if model is None:
        model = load_model()

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    emb_a = model.encode(sentences_a, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(sentences_b, normalize_embeddings=True, show_progress_bar=False)

    similarities = np.sum(emb_a * emb_b, axis=1)  # dot product từng hàng

    decisions = np.where(
        similarities < low_threshold, "stop",
        np.where(similarities > high_threshold, "early_positive", "continue")
    )

    return pd.DataFrame({
        "sentence_a": sentences_a,
        "sentence_b": sentences_b,
        "similarity": similarities,
        "decision": decisions,
    })


if __name__ == "__main__":
    result = sbert_filter(
        "The economy grew rapidly last year.",
        "Last year, the economy expanded at a fast pace.",
    )
    print(result)
