"""
build_triplet.py
-----------------
Tạo dữ liệu dạng Triplet (Anchor, Positive, Negative) để huấn luyện
Tầng 3 (Sentence-BERT fine-tuned bằng Triplet Loss).

    Anchor   = câu gốc
    Positive = câu paraphrase tương ứng (lấy thẳng từ dataset)
    Negative = 1 câu KHÔNG liên quan, được chọn theo 2 chiến lược:

        - "random"      : chọn ngẫu nhiên bất kỳ câu nào khác trong dataset
        - "hard"        : dùng TF-IDF để tìm câu có similarity CAO nhất
                          với Anchor (nhưng KHÔNG phải Positive thật của nó)
                          -> đây là "hard negative", giúp model học tốt hơn
                          vì phải phân biệt được các câu GIỐNG NHAU VỀ TỪ NGỮ
                          nhưng KHÁC NGUỒN GỐC.

Cách chạy:
    python src/data_prep/build_triplet.py --strategy hard
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import load_humarin_pairs, PROCESSED_DIR, ensure_dirs

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "triplet_data.csv")

RANDOM_SEED = 42


def _sample_random_negatives(df: pd.DataFrame) -> list:
    """
    Chiến lược đơn giản: với mỗi dòng, lấy paraphrase_text của 1 dòng
    KHÁC bất kỳ làm Negative.
    """
    rng = random.Random(RANDOM_SEED)
    n = len(df)
    negatives = []
    all_candidates = df["paraphrase_text"].tolist()

    for i in range(n):
        # Chọn ngẫu nhiên 1 index khác chính nó
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        negatives.append(all_candidates[j])

    return negatives


def _sample_hard_negatives(
    df: pd.DataFrame,
    top_k_pool: int = 5,
    chunk_size: int = 3000,
) -> list:
    """
    Chiến lược "in-batch hard negative mining" — kỹ thuật CHUẨN, được
    dùng phổ biến trong thực tế khi dataset quá lớn để tìm kiếm toàn
    cục (ví dụ: MultipleNegativesRankingLoss của sentence-transformers,
    DPR, SimCLR...).

    VẤN ĐỀ CỦA CÁCH LÀM CŨ (đã gây lỗi tràn RAM với dataset thật ~2 triệu
    dòng): tính similarity cho TOÀN BỘ N x N cặp câu cùng lúc đòi hỏi
    1 ma trận N x N phần tử — với N = 2 triệu, ma trận này cần khoảng
    32 TB RAM, KHÔNG máy tính cá nhân nào chạy nổi.

    CÁCH SỬA — chia nhỏ theo lô (chunk):
    1. Xáo trộn ngẫu nhiên toàn bộ dữ liệu 1 lần
    2. Chia thành nhiều lô nhỏ, mỗi lô `chunk_size` dòng (mặc định 3000)
    3. Trong MỖI lô, chỉ tìm hard negative trong PHẠM VI lô đó
       (ma trận similarity lúc này chỉ còn chunk_size x chunk_size,
       ví dụ 3000 x 3000 ~ 72MB, hoàn toàn an toàn về bộ nhớ)

    ĐÁNH ĐỔI cần biết: hard negative tìm được là "khó nhất TRONG LÔ
    ngẫu nhiên hiện tại", không phải "khó nhất trên TOÀN BỘ dataset".
    Đây là đánh đổi được chấp nhận rộng rãi trong thực tế nghiên cứu,
    vì tìm kiếm chính xác trên hàng triệu câu là bất khả thi về mặt
    tính toán trên phần cứng thông thường. Muốn tìm hard negative
    CHÍNH XÁC toàn cục, cần dùng thư viện tìm kiếm gần đúng chuyên
    dụng như FAISS (Facebook AI Similarity Search) — có thể nâng cấp
    sau nếu cần độ chính xác cao hơn.
    """
    rng = np.random.RandomState(RANDOM_SEED)

    n = len(df)
    anchors_all = df["original_text"].to_numpy()
    candidates_all = df["paraphrase_text"].to_numpy()

    shuffled_indices = np.arange(n)
    rng.shuffle(shuffled_indices)

    negatives = [None] * n
    num_chunks = int(np.ceil(n / chunk_size))

    for chunk_i in range(num_chunks):
        chunk_global_idx = shuffled_indices[chunk_i * chunk_size: (chunk_i + 1) * chunk_size]
        chunk_anchors = anchors_all[chunk_global_idx]
        chunk_candidates = candidates_all[chunk_global_idx]
        m = len(chunk_global_idx)

        if m <= 1:
            # Lô chỉ có 1 dòng, không có gì để so sánh -> lấy đại 1 câu
            # bất kỳ từ toàn bộ dataset làm negative (trường hợp hiếm,
            # thường chỉ xảy ra ở lô cuối cùng nếu n không chia hết)
            for local_i, global_idx in enumerate(chunk_global_idx):
                fallback_idx = rng.randrange(n)
                negatives[global_idx] = candidates_all[fallback_idx]
            continue

        vectorizer = TfidfVectorizer()
        all_texts = list(chunk_anchors) + list(chunk_candidates)
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        anchor_vecs = tfidf_matrix[:m]
        candidate_vecs = tfidf_matrix[m:]

        # Ma trận similarity ở đây chỉ có kích thước m x m (m = chunk_size,
        # ví dụ 3000 x 3000), KHÔNG PHẢI n x n -> an toàn bộ nhớ
        sim = cosine_similarity(anchor_vecs, candidate_vecs)

        # Gom nhóm original_text CHỈ trong phạm vi lô này
        local_group = {}
        for local_i, orig in enumerate(chunk_anchors):
            local_group.setdefault(orig, []).append(local_i)

        effective_top_k = min(top_k_pool, m)

        for local_i in range(m):
            row_sims = sim[local_i].copy()
            same_group_local = local_group[chunk_anchors[local_i]]
            row_sims[same_group_local] = -1

            top_k_idx = np.argsort(row_sims)[::-1][:effective_top_k]
            chosen_local = rng.choice(top_k_idx)

            global_idx = chunk_global_idx[local_i]
            negatives[global_idx] = chunk_candidates[chosen_local]

        if (chunk_i + 1) % 50 == 0 or (chunk_i + 1) == num_chunks:
            print(f"  Đã xử lý lô {chunk_i + 1}/{num_chunks} "
                  f"({(chunk_i + 1) * chunk_size:,}/{n:,} dòng)")

    return negatives


def build_triplet_dataset(strategy: str = "hard", chunk_size: int = 3000) -> pd.DataFrame:
    ensure_dirs()

    df_pairs = load_humarin_pairs()
    print(f"Đã load {len(df_pairs)} cặp (original, paraphrase)")

    if strategy == "random":
        negatives = _sample_random_negatives(df_pairs)
    elif strategy == "hard":
        negatives = _sample_hard_negatives(df_pairs, chunk_size=chunk_size)
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy} (chỉ nhận 'random' hoặc 'hard')")

    df_triplet = pd.DataFrame({
        "anchor": df_pairs["original_text"],
        "positive": df_pairs["paraphrase_text"],
        "negative": negatives,
    })

    # Kiểm tra an toàn: đảm bảo Negative không vô tình trùng Positive
    n_conflict = (df_triplet["negative"] == df_triplet["positive"]).sum()
    if n_conflict > 0:
        print(f"CẢNH BÁO: có {n_conflict} dòng Negative trùng Positive, đang loại bỏ...")
        df_triplet = df_triplet[df_triplet["negative"] != df_triplet["positive"]]

    df_triplet = df_triplet.reset_index(drop=True)
    df_triplet.to_csv(OUTPUT_PATH, index=False)
    print(f"Đã lưu {len(df_triplet)} dòng triplet vào: {OUTPUT_PATH}")

    return df_triplet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["random", "hard"], default="hard",
        help="Chiến lược chọn Negative: 'random' (nhanh) hoặc 'hard' (chất lượng cao hơn)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=3000,
        help="Kích thước mỗi lô khi tìm hard negative (chỉ áp dụng cho strategy='hard'). "
             "Giảm số này (ví dụ 1000) nếu vẫn gặp lỗi tràn RAM."
    )
    args = parser.parse_args()

    build_triplet_dataset(strategy=args.strategy, chunk_size=args.chunk_size)
