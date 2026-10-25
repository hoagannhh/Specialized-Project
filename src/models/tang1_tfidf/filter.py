"""
Tầng 1 — Lọc thô (TF-IDF + Cosine Similarity)
===============================================

VAI TRÒ ĐÚNG theo kiến trúc CaTE-X (xem sơ đồ pipeline tổng quan):
    Nhận MỘT CẶP câu (sentence_a, sentence_b) ĐÃ CÓ SẴN (từ Tầng 0 +
    Web Search, hoặc từ dataset test), tính độ tương đồng từ vựng bề
    mặt, trả điểm để hệ thống tiếp tục chạy các tầng sau, không dừng sớm.

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

DEFAULT_THRESHOLD = 0.1  # Legacy compatibility only; no early exit.


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

_vectorizer_cache = {}


def load_vectorizer(path=VECTORIZER_PATH, use_cache=True):
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path), stat.st_mtime_ns, stat.st_size
    if use_cache and key in _vectorizer_cache:
        return _vectorizer_cache[key]
    with path.open('rb') as f:
        vectorizer = pickle.load(f)
    if use_cache:
        _vectorizer_cache.clear()
        _vectorizer_cache[key] = vectorizer
    return vectorizer


def tfidf_filter_batch(pairs, vectorizer=None, threshold=None, batch_size=1024):
    """Lexical scores only. The legacy threshold does not cause early exits."""
    pairs = list(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(s,str) or not s.strip() for s in pair):
            raise ValueError('Each pair must contain two non-empty strings')
    if not pairs:
        return pd.DataFrame(columns=['sentence_a','sentence_b','similarity','decision'])
    if vectorizer is None:
        vectorizer = load_vectorizer()
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        a = vectorizer.transform([p[0] for p in batch])
        b = vectorizer.transform([p[1] for p in batch])
        scores.extend(np.asarray(a.multiply(b).sum(axis=1)).ravel().tolist())
    return pd.DataFrame({'sentence_a':[p[0] for p in pairs], 'sentence_b':[p[1] for p in pairs],
                         'similarity':scores, 'decision':'continue'})


def tfidf_filter(sentence_a, sentence_b, vectorizer=None, threshold=None):
    row = tfidf_filter_batch([(sentence_a,sentence_b)],vectorizer,batch_size=1).iloc[0]
    return {'similarity':float(row.similarity), 'decision':'continue', 'score_type':'cosine', 'reason':None}


if __name__ == '__main__':
    import argparse
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Fit tier-1 TF-IDF on training text only')
    parser.add_argument('--train', required=True, help='Triplet TRAIN CSV (never val/test)')
    parser.add_argument('--output', default=str(VECTORIZER_PATH))
    parser.add_argument('--max-sentences', type=int, default=200000)
    args = parser.parse_args()
    if args.max_sentences < 1:
        parser.error('--max-sentences must be positive')
    frame = pd.read_csv(args.train, usecols=['anchor','positive'])
    texts = pd.concat([frame.anchor,frame.positive]).dropna().drop_duplicates()
    texts = texts.sample(n=min(args.max_sentences,len(texts)),random_state=42)
    fit_reference_vectorizer(texts,save_path=Path(args.output))
