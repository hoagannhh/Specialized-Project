"""
Tầng 1 — Lọc thô (TF-IDF + Cosine Similarity)
===============================================

VAI TRÒ ĐÚNG theo kiến trúc CaTE-X (xem sơ đồ pipeline tổng quan):
    Nhận MỘT CẶP câu (sentence_a, sentence_b) ĐÃ CÓ SẴN (từ Tầng 0 +
    Web Search, hoặc từ dataset test), tính độ tương đồng từ vựng bề
    mặt, quyết định DỪNG (loại bỏ) hay TIẾP TỤC sang Tầng 2.

    Đây KHÔNG PHẢI một hệ thống tìm kiếm (retrieval) quét qua cả kho
    corpus để tìm câu giống nhất — việc "tìm ứng viên nguồn" là
    trách nhiệm của bước Web Search (nằm TRƯỚC Tầng 1 trong hệ thống
    mở rộng input-1-file), không phải của Tầng 1.

TẠI SAO PHẢI TÁCH RIÊNG BƯỚC FIT VÀ TRANSFORM:
    Nếu fit TF-IDF riêng cho từng cặp 2 câu, thống kê IDF (Inverse
    Document Frequency) gần như vô nghĩa vì "corpus" lúc đó chỉ có
    đúng 2 văn bản — đây chính là nguyên nhân gây lỗi "vocabulary chỉ
    còn lại 2 từ" ở phiên bản trước (corpus nhỏ + min_df=2 xóa sạch
    từ có nghĩa).

    Cách sửa đúng: FIT vectorizer MỘT LẦN DUY NHẤT trên 1 corpus
    THAM CHIẾU LỚN (ví dụ toàn bộ câu trong tập train của Tầng 3/4),
    lưu lại dùng nhiều lần. Lúc suy luận, KHÔNG fit lại — chỉ
    TRANSFORM 2 câu cần so sánh bằng vectorizer đã fit sẵn.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
VECTORIZER_DIR = DATA_DIR / "models" / "tang1"
VECTORIZER_PATH = VECTORIZER_DIR / "tfidf_vectorizer.pkl"

DEFAULT_THRESHOLD = 0.1  # Loại bỏ nếu similarity < ngưỡng này (đúng theo sơ đồ)


# ============================================================
# BƯỚC 1 — FIT (chỉ chạy 1 LẦN DUY NHẤT lúc chuẩn bị hệ thống)
# ============================================================

def fit_reference_vectorizer(
    reference_sentences,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    save_path: Path = VECTORIZER_PATH,
) -> TfidfVectorizer:
    """
    Fit TF-IDF vectorizer trên corpus THAM CHIẾU LỚN để có thống kê
    IDF ổn định, có ý nghĩa thống kê thật sự.

    Khuyến nghị dùng corpus tham chiếu: toàn bộ (hoặc 1 mẫu lớn, ví
    dụ 200.000-500.000 câu) từ tập train Tầng 3/4 — vì đây là corpus
    đã có sẵn, đủ lớn, và cùng domain với bài toán.

    CHỈ CẦN CHẠY HÀM NÀY 1 LẦN. Sau đó dùng load_vectorizer() ở mọi
    lần suy luận tiếp theo, không fit lại.
    """
    reference_sentences = [
        str(s).strip() for s in reference_sentences if str(s).strip()
    ]
    print(f"Fit TF-IDF trên {len(reference_sentences):,} câu tham chiếu...")

    if len(reference_sentences) < 1000:
        print(
            "CẢNH BÁO: corpus tham chiếu khá nhỏ (<1000 câu). Thống kê "
            "IDF có thể không ổn định. Cân nhắc dùng corpus lớn hơn "
            "(ví dụ toàn bộ tập train Tầng 3/4)."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(reference_sentences)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    if len(vectorizer.vocabulary_) < 100:
        print(
            "CẢNH BÁO: vocabulary quá nhỏ (<100 từ) — kiểm tra lại "
            "corpus tham chiếu hoặc giảm min_df trước khi dùng tiếp."
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Đã lưu vectorizer vào: {save_path}")

    return vectorizer


# ============================================================
# BƯỚC 2 — LOAD (dùng lại nhiều lần, có cache trong bộ nhớ)
# ============================================================

_cached_vectorizer = None  # cache cấp module, tránh đọc lại file .pkl nhiều lần


def load_vectorizer(path: Path = VECTORIZER_PATH, use_cache: bool = True) -> TfidfVectorizer:
    """
    Load vectorizer đã fit sẵn, có cache trong bộ nhớ.

    QUAN TRỌNG: nếu gọi hàm này nhiều lần trong 1 phiên chạy (ví dụ
    lặp qua hàng trăm câu trong 1 file, như kiến trúc hệ thống mở
    rộng đã thiết kế), cache giúp CHỈ ĐỌC FILE 1 LẦN DUY NHẤT, không
    tốn I/O lặp lại cho từng câu — đây chính là lỗi hiệu năng đã
    phát hiện ở phiên bản search_sentence() trước đó.
    """
    global _cached_vectorizer
    if use_cache and _cached_vectorizer is not None:
        return _cached_vectorizer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vectorizer đã fit tại: {path}\n"
            f"Hãy chạy fit_reference_vectorizer() trước (chỉ 1 lần)."
        )

    with open(path, "rb") as f:
        vectorizer = pickle.load(f)

    if use_cache:
        _cached_vectorizer = vectorizer

    return vectorizer


# ============================================================
# BƯỚC 3 — LỌC (hàm chính của Tầng 1 trong pipeline)
# ============================================================

def tfidf_filter(
    sentence_a: str,
    sentence_b: str,
    vectorizer: TfidfVectorizer = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Hàm CHÍNH của Tầng 1 — đúng vai trò trong pipeline CaTE-X.

    Input : 2 câu cụ thể (đã có sẵn, KHÔNG tìm kiếm trong corpus)
    Output: dict gồm similarity + quyết định dừng ("stop") hay
            tiếp tục ("continue") sang Tầng 2
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()

    if not sentence_a or not str(sentence_a).strip() or not sentence_b or not str(sentence_b).strip():
        return {"similarity": 0.0, "decision": "stop", "reason": "một trong hai câu rỗng"}

    vectors = vectorizer.transform([str(sentence_a), str(sentence_b)])
    # 2 vector đã chuẩn hóa L2 sẵn (TfidfVectorizer mặc định norm='l2'),
    # nên dot product = cosine similarity, không cần gọi hàm riêng
    similarity = float(vectors[0].multiply(vectors[1]).sum())

    decision = "continue" if similarity >= threshold else "stop"

    return {
        "similarity": similarity,
        "decision": decision,
        "reason": None if decision == "continue" else f"similarity < {threshold}",
    }


def tfidf_filter_batch(
    pairs: list,
    vectorizer: TfidfVectorizer = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Phiên bản xử lý HÀNG LOẠT — dùng khi cần lọc nhiều cặp câu cùng
    lúc (ví dụ: đánh giá trên tập test với hàng trăm nghìn dòng).
    Nhanh hơn nhiều so với gọi tfidf_filter() trong vòng lặp Python
    từng dòng một, vì tận dụng phép toán vector hóa của scipy.

    Tính cosine similarity TỪNG CẶP (row-wise), KHÔNG tính ma trận
    N x N đầy đủ -> an toàn bộ nhớ kể cả với hàng triệu cặp (giống
    kỹ thuật đã dùng trong build_triplet.py và run_eda.py).

    Parameters
    ----------
    pairs : list các tuple (sentence_a, sentence_b)
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()

    sentences_a = [str(p[0]) for p in pairs]
    sentences_b = [str(p[1]) for p in pairs]

    vectors_a = vectorizer.transform(sentences_a)
    vectors_b = vectorizer.transform(sentences_b)

    product = vectors_a.multiply(vectors_b)
    similarities = np.asarray(product.sum(axis=1)).flatten()

    decisions = np.where(similarities >= threshold, "continue", "stop")

    return pd.DataFrame({
        "sentence_a": sentences_a,
        "sentence_b": sentences_b,
        "similarity": similarities,
        "decision": decisions,
    })


if __name__ == "__main__":
    # ---- Demo nhanh ----
    # Bước fit (chỉ chạy 1 lần khi setup, bỏ comment khi cần):
    # fit_reference_vectorizer(reference_sentences=[...danh sách câu tham chiếu...])

    result = tfidf_filter(
        "The economy grew rapidly last year.",
        "Last year, the economy expanded quickly.",
    )
    print(result)
