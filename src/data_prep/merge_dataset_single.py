"""
merge_tang0_dataset.py

Merge:

- Humarin single-label dataset
- PADBen Task 1
- PADBen Task 2
- PADBen Task 3

Mục đích:
Tạo dataset chung cho Tầng 0 - Style Classifier.

Output:
data/processed/data_single_all.csv

Schema output:
- group_id
- sentence_position
- sentence
- label
- source

Ý nghĩa:
- Humarin:
    group_id = nhóm gồm original + các paraphrase tương ứng
    sentence_position = 0 cho original, 1+ cho paraphrase

- PADBen:
    group_id = pair_id
    sentence_position = vị trí câu trong pair

Lưu ý:
group_id chỉ dùng để chống leakage khi chia train/val/test.
Không được đưa group_id vào model như feature.
"""

import os
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

HUMARIN_PATH = os.path.join(
    PROCESSED_DIR,
    "single_labeled.csv"
)

PADBEN_DIR = os.path.join(
    PROCESSED_DIR,
    "padben"
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "data_single_all.csv"
)


# ============================================================
# LOAD HUMARIN
# ============================================================

def load_humarin():

    print("\n" + "=" * 60)
    print("LOAD HUMARIN")
    print("=" * 60)

    if not os.path.exists(HUMARIN_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{HUMARIN_PATH}"
        )

    df = pd.read_csv(HUMARIN_PATH)

    print(f"File: {HUMARIN_PATH}")
    print(f"Số dòng: {len(df)}")
    print(f"Cột: {list(df.columns)}")

    # --------------------------------------------------------
    # Kiểm tra cột bắt buộc
    # --------------------------------------------------------

    required = [
        "group_id",
        "sentence_position",
        "sentence",
        "label"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Humarin không có cột bắt buộc: {col}"
            )

    # --------------------------------------------------------
    # Chỉ giữ các cột cần thiết
    # --------------------------------------------------------

    df = df[
        [
            "group_id",
            "sentence_position",
            "sentence",
            "label"
        ]
    ].copy()

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    df["source"] = "humarin"

    return df


# ============================================================
# LOAD PADBEN
# ============================================================

def load_padben_task(task_number):

    path = os.path.join(
        PADBEN_DIR,
        f"single_task{task_number}.csv"
    )

    print("\n" + "=" * 60)
    print(f"LOAD PADBEN TASK {task_number}")
    print("=" * 60)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Số dòng: {len(df)}")
    print(f"Cột: {list(df.columns)}")

    required = [
        "pair_id",
        "sentence_position",
        "sentence",
        "label",
        "source"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"PADBen Task {task_number} "
                f"không có cột: {col}"
            )

    return df


# ============================================================
# MERGE
# ============================================================

def merge_datasets():

    # ========================================================
    # 1. LOAD HUMARIN
    # ========================================================

    humarin = load_humarin()

    # ========================================================
    # 2. LOAD PADBEN TASK 1-3
    # ========================================================

    padben_list = []

    for task in [1, 2, 3]:

        df = load_padben_task(task)

        # Đảm bảo source chính xác
        df["source"] = f"padben_task{task}"

        padben_list.append(df)

    padben = pd.concat(
        padben_list,
        ignore_index=True
    )

    # ========================================================
    # 3. CHUẨN HÓA PADBEN
    # ========================================================

    padben = padben[
        [
            "pair_id",
            "sentence_position",
            "sentence",
            "label",
            "source"
        ]
    ].copy()

    # Đổi pair_id -> group_id
    #
    # Lý do:
    # Sau khi merge, Humarin và PADBen dùng cùng một
    # metadata name là group_id.
    #
    # group_id chỉ dùng để split chống leakage.

    padben = padben.rename(
        columns={
            "pair_id": "group_id"
        }
    )

    # ========================================================
    # 4. CHUẨN HÓA HUMARIN
    # ========================================================

    humarin = humarin[
        [
            "group_id",
            "sentence_position",
            "sentence",
            "label",
            "source"
        ]
    ].copy()

    # ========================================================
    # 5. MERGE HUMARIN + PADBEN
    # ========================================================

    merged = pd.concat(
        [
            humarin,
            padben
        ],
        ignore_index=True
    )

    # ========================================================
    # 6. CLEAN SENTENCE
    # ========================================================

    merged["sentence"] = (
        merged["sentence"]
        .astype(str)
        .str.strip()
    )

    # Xóa sentence rỗng
    merged = merged[
        merged["sentence"].str.len() > 0
    ].copy()

    # ========================================================
    # 7. CLEAN LABEL
    # ========================================================

    merged["label"] = (
        pd.to_numeric(
            merged["label"],
            errors="raise"
        )
        .astype(int)
    )

    # Kiểm tra label hợp lệ
    invalid_labels = ~merged["label"].isin([0, 1])

    if invalid_labels.any():

        invalid_values = (
            merged.loc[
                invalid_labels,
                "label"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Phát hiện label không hợp lệ: "
            f"{invalid_values}"
        )

    # ========================================================
    # 8. KIỂM TRA GROUP_ID
    # ========================================================

    print("\nKiểm tra group_id...")

    print(
        f"Số group Humarin: "
        f"{humarin['group_id'].nunique()}"
    )

    print(
        f"Số group PADBen: "
        f"{padben['group_id'].nunique()}"
    )

    # ========================================================
    # 9. REMOVE DUPLICATE
    # ========================================================

    before = len(merged)
    merged = merged.drop_duplicates(
        subset=[
            "source",
            "group_id",
            "sentence_position",
            "sentence",
            "label"
        ]
    ).copy()

    after = len(merged)

    print(
        f"\nĐã xóa {before - after} dòng duplicate."
    )

    # ========================================================
    # 10. SHUFFLE
    # ========================================================

    merged = merged.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ========================================================
    # 11. SAVE
    # ========================================================

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 12. REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("MERGE HOÀN TẤT")
    print("=" * 60)

    print(f"Output: {OUTPUT_PATH}")

    print(
        f"Tổng số dòng: {len(merged)}"
    )

    print("\nCột:")
    print(
        list(merged.columns)
    )

    print("\nPhân bố source:")
    print(
        merged["source"]
        .value_counts()
    )

    print("\nPhân bố label:")
    print(
        merged["label"]
        .value_counts()
    )

    print("\nSource x Label:")
    print(
        pd.crosstab(
            merged["source"],
            merged["label"]
        )
    )

    print("\nSố group theo source:")

    print(
        merged.groupby("source")["group_id"]
        .nunique()
    )

    print("\n5 dòng đầu:")
    print(
        merged.head()
        .to_string(index=False)
    )

    return merged


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    raise SystemExit("Legacy pipeline retired. Run: python src/data_prep/prepare_datasets.py --strategy mixed")
