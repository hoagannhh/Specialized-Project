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
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from common import PROCESSED_DIR, SPLITS_DIR, ensure_dirs

RANDOM_SEED = 42


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
    args = parser.parse_args()

    split_dataset(
        input_filename=args.input,
        val_size=args.val_size,
        test_size=args.test_size,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
    )
