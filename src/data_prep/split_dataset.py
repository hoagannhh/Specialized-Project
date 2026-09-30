"""
split_dataset.py
-----------------
Chia 1 file dữ liệu đã xử lý (trong data/processed/) thành 3 tập:
train / val / test, theo tỷ lệ 70/15/15 (mặc định).

Dùng chung được cho cả 3 loại dữ liệu:
    - pairwise_labeled.csv  (có cột 'label' -> chia theo kiểu stratify
                             để giữ đúng tỷ lệ 2 lớp ở cả 3 tập)
    - single_labeled.csv    (tương tự, có cột 'label')
    - triplet_data.csv      (KHÔNG có cột label -> chia ngẫu nhiên thường)

QUAN TRỌNG — tránh rò rỉ dữ liệu (data leakage):
    Với triplet_data.csv, cần đảm bảo các dòng có CÙNG 'anchor'
    không bị chia vào 2 tập khác nhau (ví dụ 1 câu gốc vừa xuất hiện
    ở train vừa xuất hiện ở test) — vì như vậy model có thể "nhìn thấy"
    gián tiếp câu đó lúc train, khiến đánh giá trên test không còn
    khách quan. Hàm split_by_group() bên dưới xử lý đúng vấn đề này.

Cách chạy:
    python src/data_prep/split_dataset.py --input triplet_data.csv --group-col anchor
    python src/data_prep/split_dataset.py --input pairwise_labeled.csv --stratify-col label
    python src/data_prep/split_dataset.py --input single_labeled.csv --stratify-col label
    python src/data_prep/split_dataset.py --input data_single_all.csv --humarin
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from common import PROCESSED_DIR, SPLITS_DIR, ensure_dirs

RANDOM_SEED = 42


def split_tang0_by_group(
    df,
    group_col="group_id",
    source_col="source",
    val_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Split toàn bộ dataset Tầng 0 theo group.

    Dataset gồm:
        - Humarin
        - PADBen Task 1
        - PADBen Task 2
        - PADBen Task 3

    Quy tắc:
        - Các sentence cùng một group phải nằm cùng một split.
        - Group được xác định bởi:
              (source, group_id)

    Ví dụ:

        Humarin group_id = 100
        PADBen task1 group_id = 100

    Hai group này KHÁC nhau vì source khác nhau.

    Returns
    -------
    train_df
    val_df
    test_df
    """

    # ======================================================
    # 1. Kiểm tra input
    # ======================================================

    required_cols = [
        group_col,
        source_col
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing_cols}"
        )

    # ======================================================
    # 2. Kiểm tra group_id
    # ======================================================

    if df[group_col].isna().any():

        n_missing = df[group_col].isna().sum()

        raise ValueError(
            f"Phát hiện {n_missing:,} dòng có "
            f"{group_col} bị thiếu."
        )

    if df[source_col].isna().any():

        n_missing = df[source_col].isna().sum()

        raise ValueError(
            f"Phát hiện {n_missing:,} dòng có "
            f"{source_col} bị thiếu."
        )

    # ======================================================
    # 3. Kiểm tra source
    # ======================================================

    print("\n" + "=" * 60)
    print("TẦNG 0 - GROUP-AWARE SPLIT")
    print("=" * 60)

    print(
        f"Tổng số samples: {len(df):,}"
    )

    print("\nPhân bố source:")
    print(
        df[source_col]
        .value_counts()
    )

    # ======================================================
    # 4. Tạo global group key
    # ======================================================
    #
    # Không được chỉ dùng group_id.
    #
    # Vì:
    #
    # Humarin group_id = 100
    # PADBen group_id  = 100
    #
    # là 2 group khác nhau.
    #
    # Do đó dùng:
    #
    # source + group_id
    #
    # ======================================================

    df = df.copy()

    df["_split_group"] = (
        df[source_col].astype(str)
        + "___"
        + df[group_col].astype(str)
    )

    # ======================================================
    # 5. Kiểm tra số group
    # ======================================================

    n_groups = df["_split_group"].nunique()

    print(
        f"\nTổng số groups: {n_groups:,}"
    )

    print("\nSố groups theo source:")

    groups_by_source = (
        df.groupby(source_col)["_split_group"]
        .nunique()
    )

    print(groups_by_source)

    # ======================================================
    # 6. Shuffle groups
    # ======================================================

    groups = (
        df["_split_group"]
        .drop_duplicates()
        .sample(
            frac=1,
            random_state=random_state
        )
        .tolist()
    )

    # ======================================================
    # 7. Tính số group
    # ======================================================

    n_test = round(
        n_groups * test_size
    )

    n_val = round(
        n_groups * val_size
    )

    n_train = (
        n_groups
        - n_test
        - n_val
    )

    print("\nPhân chia groups:")

    print(
        f"Train groups: {n_train:,}"
    )

    print(
        f"Val groups:   {n_val:,}"
    )

    print(
        f"Test groups:  {n_test:,}"
    )

    # ======================================================
    # 8. Tạo group sets
    # ======================================================

    test_groups = set(
        groups[:n_test]
    )

    val_groups = set(
        groups[
            n_test:
            n_test + n_val
        ]
    )

    train_groups = set(
        groups[
            n_test + n_val:
        ]
    )

    # ======================================================
    # 9. Tạo train / val / test
    # ======================================================

    train_df = df[
        df["_split_group"].isin(
            train_groups
        )
    ].copy()

    val_df = df[
        df["_split_group"].isin(
            val_groups
        )
    ].copy()

    test_df = df[
        df["_split_group"].isin(
            test_groups
        )
    ].copy()

    # ======================================================
    # 10. Xóa cột internal
    # ======================================================

    train_df = train_df.drop(
        columns=["_split_group"]
    )

    val_df = val_df.drop(
        columns=["_split_group"]
    )

    test_df = test_df.drop(
        columns=["_split_group"]
    )

    # ======================================================
    # 11. Shuffle từng split
    # ======================================================

    train_df = (
        train_df
        .sample(
            frac=1,
            random_state=random_state
        )
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .sample(
            frac=1,
            random_state=random_state
        )
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sample(
            frac=1,
            random_state=random_state
        )
        .reset_index(drop=True)
    )

    # ======================================================
    # 12. KIỂM TRA LEAKAGE
    # ======================================================

    def get_group_set(split_df):

        return set(
            zip(
                split_df[source_col].astype(str),
                split_df[group_col].astype(str)
            )
        )

    train_groups_check = get_group_set(
        train_df
    )

    val_groups_check = get_group_set(
        val_df
    )

    test_groups_check = get_group_set(
        test_df
    )

    train_val_overlap = (
        train_groups_check
        & val_groups_check
    )

    train_test_overlap = (
        train_groups_check
        & test_groups_check
    )

    val_test_overlap = (
        val_groups_check
        & test_groups_check
    )

    # ======================================================
    # 13. In kết quả leakage
    # ======================================================

    print("\n" + "=" * 60)
    print("KIỂM TRA GROUP LEAKAGE")
    print("=" * 60)

    print(
        f"Train ∩ Val:  "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Val ∩ Test:   "
        f"{len(val_test_overlap)}"
    )

    if (
        len(train_val_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(val_test_overlap) == 0
    ):

        print(
            "\n✅ Không phát hiện group leakage."
        )

    else:

        raise RuntimeError(
            "\n❌ Phát hiện group leakage!"
        )

    # ======================================================
    # 14. Kiểm tra tổng số samples
    # ======================================================

    total_split = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    print("\nKiểm tra số lượng:")

    print(
        f"Original: "
        f"{len(df):,}"
    )

    print(
        f"Train + Val + Test: "
        f"{total_split:,}"
    )

    if total_split != len(df):

        raise RuntimeError(
            "Số lượng samples sau split "
            "không khớp với dataset ban đầu."
        )

    print(
        "✅ Không mất sample trong quá trình split."
    )

    # ======================================================
    # 15. Thống kê từng split
    # ======================================================

    print("\n" + "=" * 60)
    print("KẾT QUẢ SPLIT")
    print("=" * 60)

    for name, split_df in [
        ("TRAIN", train_df),
        ("VAL", val_df),
        ("TEST", test_df)
    ]:

        print(f"\n{name}")

        print(
            f"Samples: "
            f"{len(split_df):,}"
        )

        print(
            f"Groups:  "
            f"{len(get_group_set(split_df)):,}"
        )

        print("\nSource:")

        print(
            split_df[source_col]
            .value_counts()
        )

        if "label" in split_df.columns:

            print("\nLabel:")

            label_counts = (
                split_df["label"]
                .value_counts()
                .sort_index()
            )

            for label, count in label_counts.items():

                percentage = (
                    count
                    / len(split_df)
                    * 100
                )

                print(
                    f"  Label {label}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

    # ======================================================
    # 16. Trả về
    # ======================================================

    return (
        train_df,
        val_df,
        test_df
    )

def split_humarin_merged(
    df,
    group_col="group_id",
    source_col="source",
    source_name="humarin",
    val_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Chia riêng dữ liệu Humarin theo group_id.

    Chỉ các dòng có:
        source == "humarin"

    được đưa vào quá trình split.

    Toàn bộ các sentence có cùng group_id
    sẽ nằm trong cùng một split.

    PADBen và các source khác không được xử lý
    bởi hàm này.

    Returns
    -------
    train_df
    val_df
    test_df
    """

    # ======================================================
    # 1. Kiểm tra cột
    # ======================================================

    required_cols = [
        group_col,
        source_col
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing_cols}"
        )

    # ======================================================
    # 2. Lấy riêng Humarin
    # ======================================================

    df_humarin = df[
        df[source_col] == source_name
    ].copy()

    if len(df_humarin) == 0:
        raise ValueError(
            f"Không tìm thấy dữ liệu source='{source_name}'"
        )

    # Kiểm tra group_id
    if df_humarin[group_col].isna().any():
        raise ValueError(
            "Humarin đang có group_id bị thiếu. "
            "Hãy kiểm tra lại build_single_label.py."
        )

    print("\n===== HUMARIN GROUP SPLIT =====")

    print(
        f"Tổng số samples: {len(df_humarin):,}"
    )

    print(
        f"Số groups: "
        f"{df_humarin[group_col].nunique():,}"
    )

    # ======================================================
    # 3. Shuffle danh sách group
    # ======================================================

    groups = (
        df_humarin[group_col]
        .drop_duplicates()
        .sample(
            frac=1,
            random_state=random_state
        )
        .tolist()
    )

    n_groups = len(groups)

    # ======================================================
    # 4. Tính số group cho từng tập
    # ======================================================

    n_test = round(
        n_groups * test_size
    )

    n_val = round(
        n_groups * val_size
    )

    # ======================================================
    # 5. Chia group
    # ======================================================

    test_groups = set(
        groups[:n_test]
    )

    val_groups = set(
        groups[
            n_test:
            n_test + n_val
        ]
    )

    train_groups = set(
        groups[
            n_test + n_val:
        ]
    )

    # ======================================================
    # 6. Tạo dataset
    # ======================================================

    train_df = df_humarin[
        df_humarin[group_col].isin(
            train_groups
        )
    ].copy()

    val_df = df_humarin[
        df_humarin[group_col].isin(
            val_groups
        )
    ].copy()

    test_df = df_humarin[
        df_humarin[group_col].isin(
            test_groups
        )
    ].copy()

    # ======================================================
    # 7. Shuffle từng split
    # ======================================================

    train_df = train_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    # ======================================================
    # 8. Kiểm tra leakage
    # ======================================================

    train_groups_check = set(
        train_df[group_col]
    )

    val_groups_check = set(
        val_df[group_col]
    )

    test_groups_check = set(
        test_df[group_col]
    )

    train_val_overlap = (
        train_groups_check &
        val_groups_check
    )

    train_test_overlap = (
        train_groups_check &
        test_groups_check
    )

    val_test_overlap = (
        val_groups_check &
        test_groups_check
    )

    print("\n===== KIỂM TRA GROUP LEAKAGE =====")

    print(
        f"Train ∩ Val:  "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Val ∩ Test:   "
        f"{len(val_test_overlap)}"
    )

    if (
        len(train_val_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(val_test_overlap) == 0
    ):
        print(
            "✅ Không phát hiện group leakage."
        )
    else:
        raise RuntimeError(
            "❌ Phát hiện group leakage!"
        )

    # ======================================================
    # 9. In thống kê
    # ======================================================

    print("\n===== KẾT QUẢ SPLIT =====")

    for name, split_df in [
        ("TRAIN", train_df),
        ("VAL", val_df),
        ("TEST", test_df)
    ]:

        print(
            f"\n{name}:"
        )

        print(
            f"  Samples: "
            f"{len(split_df):,}"
        )

        print(
            f"  Groups:  "
            f"{split_df[group_col].nunique():,}"
        )

        if "label" in split_df.columns:

            label_counts = (
                split_df["label"]
                .value_counts()
                .sort_index()
            )

            print(
                f"  Label distribution:"
            )

            for label, count in label_counts.items():

                percentage = (
                    count /
                    len(split_df) *
                    100
                )

                print(
                    f"    Label {label}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

    return (
        train_df,
        val_df,
        test_df
    )

def split_by_group(df: pd.DataFrame, group_col: str,
                    val_size: float, test_size: float):
    """
    Chia dữ liệu sao cho các dòng có cùng giá trị ở group_col
    (ví dụ cùng 1 câu 'anchor') LUÔN nằm trong CÙNG 1 tập
    (không bị tách rải rác qua cả train/val/test).
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss1.split(df, groups=df[group_col]))

    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    # Tỷ lệ val tính lại trên phần còn lại (train_val) cho đúng tỷ lệ tổng
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val[group_col]))

    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    return df_train, df_val, df_test


def split_by_stratify(df: pd.DataFrame, stratify_col: str,
                       val_size: float, test_size: float):
    """Chia ngẫu nhiên nhưng giữ đúng tỷ lệ các lớp (label) ở mỗi tập."""
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED,
        stratify=df[stratify_col],
    )
    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val_size, random_state=RANDOM_SEED,
        stratify=df_train_val[stratify_col],
    )
    return df_train, df_val, df_test


def split_plain(df: pd.DataFrame, val_size: float, test_size: float):
    """Chia ngẫu nhiên thuần túy, không nhóm cũng không giữ tỷ lệ lớp."""
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED
    )
    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val_size, random_state=RANDOM_SEED
    )
    return df_train, df_val, df_test


def split_dataset(
    input_filename: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    group_col: str = None,
    stratify_col: str = None,
):
    ensure_dirs()

    input_path = os.path.join(PROCESSED_DIR, input_filename)
    df = pd.read_csv(input_path)
    print(f"Đã đọc {len(df)} dòng từ {input_path}")

    if group_col:
        df_train, df_val, df_test = split_by_group(df, group_col, val_size, test_size)
        print(f"Chia theo group '{group_col}' (tránh rò rỉ dữ liệu)")
    elif stratify_col:
        df_train, df_val, df_test = split_by_stratify(df, stratify_col, val_size, test_size)
        print(f"Chia theo stratify '{stratify_col}' (giữ tỷ lệ lớp)")
    else:
        df_train, df_val, df_test = split_plain(df, val_size, test_size)
        print("Chia ngẫu nhiên thuần túy (không group, không stratify)")

    base_name = os.path.splitext(input_filename)[0]
    train_path = os.path.join(SPLITS_DIR, f"{base_name}_train.csv")
    val_path = os.path.join(SPLITS_DIR, f"{base_name}_val.csv")
    test_path = os.path.join(SPLITS_DIR, f"{base_name}_test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"  train: {len(df_train):>6} dòng -> {train_path}")
    print(f"  val:   {len(df_val):>6} dòng -> {val_path}")
    print(f"  test:  {len(df_test):>6} dòng -> {test_path}")

    # Kiểm tra an toàn nếu có group_col: đảm bảo không group nào bị lặp
    # ở nhiều hơn 1 tập
    if group_col:
        train_groups = set(df_train[group_col])
        val_groups = set(df_val[group_col])
        test_groups = set(df_test[group_col])
        overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
        if overlap:
            print(f"CẢNH BÁO: phát hiện {len(overlap)} group bị trùng giữa các tập! "
                  f"Có nguy cơ rò rỉ dữ liệu, cần kiểm tra lại.")
        else:
            print("Đã xác nhận: không có group nào bị trùng giữa train/val/test.")

    return df_train, df_val, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Tên file trong data/processed/, ví dụ triplet_data.csv")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--group-col", default=None, help="Cột dùng để nhóm, ví dụ 'anchor'")
    parser.add_argument("--stratify-col", default=None, help="Cột dùng để giữ tỷ lệ lớp, ví dụ 'label'")
    parser.add_argument("--humarin", action="store_true", help="Split riêng Humarin theo group_id")
    parser.add_argument("--tang0", action="store_true", help="Split dataset Tầng 0 theo source + group_id")
    args = parser.parse_args()

    # split_dataset(
    #     input_filename=args.input,
    #     val_size=args.val_size,
    #     test_size=args.test_size,
    #     group_col=args.group_col,
    #     stratify_col=args.stratify_col,
    # )
    if args.tang0:

        input_path = os.path.join(
            PROCESSED_DIR,
            args.input
        )

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Không tìm thấy file:\n{input_path}"
            )

        print("\n" + "=" * 60)
        print("LOAD DATASET TẦNG 0")
        print("=" * 60)

        df = pd.read_csv(input_path)

        train_df, val_df, test_df = split_tang0_by_group(
            df=df,
            group_col="group_id",
            source_col="source",
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=42
        )

        # ======================================================
        # SAVE
        # ======================================================

        train_path = os.path.join(
            SPLITS_DIR,
            "tang0_train.csv"
        )

        val_path = os.path.join(
            SPLITS_DIR,
            "tang0_val.csv"
        )

        test_path = os.path.join(
            SPLITS_DIR,
            "tang0_test.csv"
        )

        train_df.to_csv(
            train_path,
            index=False,
            encoding="utf-8-sig"
        )

        val_df.to_csv(
            val_path,
            index=False,
            encoding="utf-8-sig"
        )

        test_df.to_csv(
            test_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 60)
        print("ĐÃ LƯU DATASET TẦNG 0")
        print("=" * 60)

        print(f"Train: {train_path}")
        print(f"Val:   {val_path}")
        print(f"Test:  {test_path}")
 
    else:

        split_dataset(
            input_filename=args.input,
            val_size=args.val_size,
            test_size=args.test_size,
            group_col=args.group_col,
            stratify_col=args.stratify_col,
        )