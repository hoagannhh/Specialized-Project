"""
convert_padben_format.py
-------------------------
Chuyển dataset PADBen (đã tải bằng download_padben.py) về cùng định
dạng (sentence_1, sentence_2, label) với pairwise_labeled.csv, để có
thể đưa thẳng vào evaluation/cross_dataset_eval.py mà không cần sửa
code đánh giá.

*** CẢNH BÁO QUAN TRỌNG — CẦN BẠN TỰ KIỂM TRA TRƯỚC KHI DÙNG ***
Tên cột thực tế của PADBen (`sentence_pair`, `label_pair`, hoặc tên
khác tùy config `task1`...`task5`) CHƯA được xác minh đầy đủ ở đây.
Hàm `inspect_columns()` bên dưới sẽ in ra cấu trúc THẬT của file bạn
tải về — HÃY CHẠY HÀM NÀY TRƯỚC, đọc kỹ output, rồi mới chỉnh lại
COLUMN_MAPPING cho khớp với dữ liệu thật, TRƯỚC KHI chạy convert().

Cách chạy:
    # Bước 1 — bắt buộc chạy trước để xem cấu trúc thật:
    python src/data_prep/convert_padben_format.py --inspect

    # Bước 2 — sau khi đã chỉnh COLUMN_MAPPING cho khớp, mới chạy:
    python src/data_prep/convert_padben_format.py --convert
"""

import argparse
import json
import os

import pandas as pd

from common import RAW_DIR, ensure_dirs

INPUT_PATH = os.path.join(RAW_DIR, "padben_sentence_pair_task1.json")
OUTPUT_PATH = os.path.join(
    os.path.dirname(RAW_DIR), "cross_eval", "padben_test_processed.csv"
)

# !!! CHỈNH LẠI 3 GIÁ TRỊ NÀY CHO KHỚP VỚI TÊN CỘT THẬT SAU KHI
# !!! CHẠY --inspect VÀ ĐỌC KỸ OUTPUT
COLUMN_MAPPING = {
    "sentence_1": "sentence_pair_1",   # TODO: sửa lại cho khớp dữ liệu thật
    "sentence_2": "sentence_pair_2",   # TODO: sửa lại cho khớp dữ liệu thật
    "label": "label_pair",             # TODO: sửa lại cho khớp dữ liệu thật
}


def inspect_columns(input_path: str = INPUT_PATH):
    """
    In ra cấu trúc THẬT của file PADBen đã tải về — chạy hàm này
    TRƯỚC khi chỉnh COLUMN_MAPPING, đừng đoán mò tên cột.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {input_path}. "
            f"Hãy chạy download_padben.py trước."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    print(f"Số dòng: {len(df)}")
    print(f"Các cột hiện có: {list(df.columns)}")
    print("--- 3 dòng đầu tiên (xem kỹ để hiểu ý nghĩa từng cột) ---")
    for i in range(min(3, len(df))):
        print(f"\n[Dòng {i}]")
        for col in df.columns:
            value_preview = str(df.iloc[i][col])[:150]
            print(f"  {col}: {value_preview}")

    return df


def convert_padben(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    column_mapping: dict = None,
):
    ensure_dirs()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    column_mapping = column_mapping or COLUMN_MAPPING

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df_raw = pd.DataFrame(records)

    missing_cols = [v for v in column_mapping.values() if v not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"Các cột sau KHÔNG tồn tại trong dữ liệu PADBen: {missing_cols}.\n"
            f"Các cột thật sự có: {list(df_raw.columns)}.\n"
            f"Hãy chạy lại với --inspect và sửa COLUMN_MAPPING cho đúng "
            f"trước khi convert."
        )

    df_converted = pd.DataFrame({
        "sentence_1": df_raw[column_mapping["sentence_1"]],
        "sentence_2": df_raw[column_mapping["sentence_2"]],
        "label": df_raw[column_mapping["label"]],
    })

    n_before = len(df_converted)
    df_converted = df_converted.dropna().reset_index(drop=True)
    n_after = len(df_converted)
    if n_before != n_after:
        print(f"Đã loại {n_before - n_after} dòng thiếu dữ liệu.")

    # Kiểm tra nhãn có đúng dạng nhị phân (0/1) như pairwise_labeled.csv không
    unique_labels = sorted(df_converted["label"].unique().tolist())
    if unique_labels not in ([0, 1], [0], [1]):
        print(
            f"CẢNH BÁO: nhãn trong PADBen là {unique_labels}, không phải "
            f"dạng nhị phân [0, 1] thuần túy như pairwise_labeled.csv. "
            f"Cần thêm bước ánh xạ nhãn (label mapping) trước khi dùng "
            f"chung với evaluate_full_pipeline.py."
        )

    df_converted.to_csv(output_path, index=False)
    print(f"Đã lưu {len(df_converted)} dòng vào: {output_path}")

    return df_converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Chỉ in cấu trúc cột thật, KHÔNG convert")
    parser.add_argument("--convert", action="store_true",
                         help="Thực hiện convert (chỉ chạy SAU khi đã inspect và sửa COLUMN_MAPPING)")
    args = parser.parse_args()

    if args.inspect:
        inspect_columns()
    elif args.convert:
        convert_padben()
    else:
        print("Cần chỉ định --inspect (xem cấu trúc trước) hoặc --convert (chuyển đổi).")
        print("Khuyến nghị: luôn chạy --inspect trước tiên.")
