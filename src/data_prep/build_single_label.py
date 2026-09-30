"""
build_single_label.py
----------------------
Tạo dữ liệu dạng đơn-câu có nhãn (sentence, label) để huấn luyện
Bước 0 (bộ phân loại "câu này do người viết hay đã qua AI paraphrase").

    label = 0  ->  câu original_text (giả định là văn bản gốc, "người viết")
    label = 1  ->  câu paraphrase_text (do ChatGPT sinh ra, "AI-touched")

LƯU Ý QUAN TRỌNG VỀ GIẢ ĐỊNH:
    Dataset humarin/chatgpt-paraphrases lấy nguồn "text" gốc từ Quora,
    SQuAD, CNN news — đây là văn bản do CON NGƯỜI viết (không phải AI).
    Vì vậy gán nhãn 0 cho original_text là hợp lý. Nếu sau này đổi
    sang dataset khác, cần kiểm tra lại giả định này có còn đúng không.

    Mỗi original_text bị lặp lại nhiều lần (vì có nhiều paraphrase),
    nên cần loại bỏ trùng lặp Ở PHÍA original_text trước khi gộp,
    tránh việc 1 câu gốc bị đếm là nhiều mẫu label=0 giống hệt nhau.

Cách chạy:
    python src/data_prep/build_single_label.py
"""

"""
build_single_label.py

Tạo dataset cho Tầng 0:

0 = Human / Original
1 = AI-paraphrased

Mỗi original_text và toàn bộ paraphrases
của nó dùng chung group_id.
"""

import os
import pandas as pd

from common import (
    load_humarin_pairs,
    PROCESSED_DIR,
    ensure_dirs
)


OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "single_labeled.csv"
)

RANDOM_SEED = 42


def build_single_label_dataset() -> pd.DataFrame:

    ensure_dirs()

    # ==========================================================
    # 1. Load Humarin
    # ==========================================================

    df_pairs = load_humarin_pairs()

    print(
        f"Đã load {len(df_pairs):,} cặp "
        f"original-paraphrase"
    )

    print(
        f"Số group Humarin: "
        f"{df_pairs['group_id'].nunique():,}"
    )

    # ==========================================================
    # 2. HUMAN
    # ==========================================================

    df_human = (
        df_pairs[
            [
                "group_id",
                "original_text"
            ]
        ]
        .drop_duplicates(
            subset=[
                "group_id",
                "original_text"
            ]
        )
        .rename(
            columns={
                "original_text": "sentence"
            }
        )
    )

    df_human["sentence_position"] = 0
    df_human["label"] = 0
    df_human["source"] = "humarin"

    # ==========================================================
    # 3. AI PARAPHRASE
    # ==========================================================

    df_ai = (
        df_pairs[
            [
                "group_id",
                "paraphrase_text"
            ]
        ]
        .drop_duplicates(
            subset=[
                "group_id",
                "paraphrase_text"
            ]
        )
        .rename(
            columns={
                "paraphrase_text": "sentence"
            }
        )
    )

    df_ai["label"] = 1
    df_ai["source"] = "humarin"

    # Mỗi paraphrase được đánh position riêng
    df_ai["sentence_position"] = (
        df_ai
        .groupby("group_id")
        .cumcount()
        + 1
    )

    # ==========================================================
    # 4. Merge Human + AI
    # ==========================================================

    df_final = pd.concat(
        [
            df_human,
            df_ai
        ],
        ignore_index=True
    )

    # ==========================================================
    # 5. Chuẩn hóa
    # ==========================================================

    df_final["sentence"] = (
        df_final["sentence"]
        .astype(str)
        .str.strip()
    )

    df_final = df_final[
        df_final["sentence"].str.len() > 0
    ]

    # ==========================================================
    # 6. Loại duplicate thật sự
    # ==========================================================

    df_final = df_final.drop_duplicates(
        subset=[
            "group_id",
            "sentence_position",
            "sentence",
            "label"
        ]
    )

    # ==========================================================
    # 7. Shuffle
    # ==========================================================

    df_final = (
        df_final
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    # ==========================================================
    # 8. Đưa column về đúng thứ tự
    # ==========================================================

    df_final = df_final[
        [
            "group_id",
            "sentence_position",
            "sentence",
            "label",
            "source"
        ]
    ]

    # ==========================================================
    # 9. Statistics
    # ==========================================================

    print("\n===== THỐNG KÊ =====")

    print(
        f"Tổng samples: "
        f"{len(df_final):,}"
    )

    print(
        f"Tổng groups: "
        f"{df_final['group_id'].nunique():,}"
    )

    print("\nLabel:")
    print(df_final["label"].value_counts())

    print("\nSố sample/group:")
    print(
        df_final
        .groupby("group_id")
        .size()
        .value_counts()
        .sort_index()
    )

    # ==========================================================
    # 10. Save
    # ==========================================================

    df_final.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nĐã lưu dataset tại:\n"
        f"{OUTPUT_PATH}"
    )

    return df_final


if __name__ == "__main__":
    build_single_label_dataset()