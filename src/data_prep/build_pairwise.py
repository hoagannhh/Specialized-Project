"""
build_pairwise.py
------------------
Tạo dữ liệu dạng Cặp-có-nhãn (sentence_1, sentence_2, label) để huấn
luyện Tầng 4 (Cross-Encoder + Binary Cross-Entropy).

    label = 1  ->  cặp câu THẬT SỰ là (gốc, paraphrase) — lấy trực tiếp
                   từ dataset
    label = 0  ->  cặp câu KHÔNG liên quan — ghép ngẫu nhiên 2 câu
                   không cùng nguồn gốc

Để dữ liệu CÂN BẰNG (balanced), số lượng mẫu label=0 được tạo ra
BẰNG với số lượng mẫu label=1 (tỷ lệ 1:1) — tránh model học lệch
về phía 1 lớp chiếm đa số.

Cách chạy:
    python src/data_prep/build_pairwise.py
"""

import os
import random

import pandas as pd

from common import load_humarin_pairs, PROCESSED_DIR, ensure_dirs

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "pairwise_labeled.csv")
RANDOM_SEED = 42


def _build_positive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sentence_1": df["original_text"],
        "sentence_2": df["paraphrase_text"],
        "label": 1,
    })


def _build_negative_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ghép ngẫu nhiên: sentence_1 lấy từ original_text của dòng i,
    sentence_2 lấy từ paraphrase_text của dòng j (i != j VÀ 2 dòng
    không cùng nhóm original_text, để tránh vô tình ghép đúng 1 cặp
    thật sự liên quan).
    """
    rng = random.Random(RANDOM_SEED)
    n = len(df)

    originals = df["original_text"].tolist()
    candidates = df["paraphrase_text"].tolist()

    original_to_indices = {}
    for idx, orig in enumerate(originals):
        original_to_indices.setdefault(orig, []).append(idx)

    sentence_1_list = []
    sentence_2_list = []

    for i in range(n):
        same_group = set(original_to_indices[originals[i]])
        j = rng.randrange(n)
        # Thử lại nếu vô tình chọn trúng dòng cùng nhóm (cùng nguồn gốc)
        max_retry = 20
        retry = 0
        while j in same_group and retry < max_retry:
            j = rng.randrange(n)
            retry += 1

        sentence_1_list.append(originals[i])
        sentence_2_list.append(candidates[j])

    return pd.DataFrame({
        "sentence_1": sentence_1_list,
        "sentence_2": sentence_2_list,
        "label": 0,
    })


def build_pairwise_dataset() -> pd.DataFrame:
    ensure_dirs()

    df_pairs = load_humarin_pairs()
    print(f"Đã load {len(df_pairs)} cặp gốc để xây dựng dữ liệu Tầng 4")

    df_positive = _build_positive_pairs(df_pairs)
    df_negative = _build_negative_pairs(df_pairs)

    df_final = pd.concat([df_positive, df_negative], ignore_index=True)

    # Trộn ngẫu nhiên thứ tự các dòng (tránh dồn hết positive lên trên)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Tổng số cặp: {len(df_final)} "
          f"(label=1: {(df_final['label'] == 1).sum()}, "
          f"label=0: {(df_final['label'] == 0).sum()})")

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Đã lưu vào: {OUTPUT_PATH}")

    return df_final


if __name__ == "__main__":
    build_pairwise_dataset()
