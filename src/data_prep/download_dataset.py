"""
download_dataset.py
--------------------
Tải dataset chính `humarin/chatgpt-paraphrases` từ HuggingFace về máy,
lưu thành file CSV trong data/raw/.

LƯU Ý QUAN TRỌNG:
    Script này cần kết nối internet tới HuggingFace Hub.
    Chạy trên Google Colab / Kaggle Notebook, KHÔNG chạy được trong môi
    trường sandbox không có internet.

Cách chạy:
    python src/data_prep/download_dataset.py
"""

import os
from datasets import load_dataset

from common import RAW_DIR, HUMARIN_RAW_PATH, ensure_dirs


def download_humarin_dataset(output_path: str = HUMARIN_RAW_PATH) -> None:
    ensure_dirs()

    print("Đang tải dataset 'humarin/chatgpt-paraphrases' từ HuggingFace...")
    dataset = load_dataset("humarin/chatgpt-paraphrases")

    # Dataset này chỉ có 1 split "train", chưa chia sẵn train/val/test
    # (việc chia nhóm nguồn do prepare_datasets.py đảm nhiệm ở bước sau)
    df = dataset["train"].to_pandas()

    print(f"Đã tải xong: {len(df)} dòng dữ liệu gốc")
    print(f"Các cột hiện có: {list(df.columns)}")

    if "text" not in df.columns or "paraphrases" not in df.columns:
        print(
            "CẢNH BÁO: cấu trúc cột không khớp với dự kiến "
            "('text', 'paraphrases'). Cần kiểm tra lại và chỉnh sửa "
            "hàm load_humarin_pairs() trong common.py cho phù hợp."
        )

    df.to_csv(output_path, index=False)
    print(f"Đã lưu vào: {output_path}")


if __name__ == "__main__":
    download_humarin_dataset()
