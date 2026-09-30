"""
common.py
---------
Các hàm dùng chung cho toàn bộ script trong thư mục data_prep.

Dataset gốc `humarin/chatgpt-paraphrases` có cấu trúc:
    - text        : câu văn bản gốc
    - paraphrases : MỘT DANH SÁCH các câu paraphrase, được lưu dưới dạng
                    chuỗi (string) đại diện cho list Python, ví dụ:
                    "['Câu paraphrase 1', 'Câu paraphrase 2', ...]"
    - category    : chủ đề nguồn (quora / squad / cnn_news)
    - source      : nguồn cụ thể

Vì cột `paraphrases` chứa NHIỀU câu paraphrase cho 1 câu gốc, ta cần
"explode" (tách) mỗi paraphrase thành 1 dòng riêng để dễ xử lý tiếp
ở các bước sau (build_triplet, build_pairwise, build_single_label).
"""

import ast
import os
import pandas as pd

# Đường dẫn thư mục gốc của project (tự động dò từ vị trí file này)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

HUMARIN_RAW_PATH = os.path.join(RAW_DIR, "humarin_chatgpt_paraphrases.csv")


def _parse_paraphrase_list(raw_value):
    """
    Chuyển cột 'paraphrases' (dạng chuỗi "['a', 'b']") thành list Python thật.

    Có 2 khả năng xảy ra tùy phiên bản dataset:
    1. Giá trị đã là list Python sẵn (nếu đọc từ parquet/HF datasets trực tiếp)
    2. Giá trị là chuỗi text cần parse bằng ast.literal_eval (nếu đọc từ CSV)

    Dùng ast.literal_eval thay vì eval() để AN TOÀN hơn — không thực thi
    mã tùy ý, chỉ parse cấu trúc dữ liệu Python cơ bản (list, string, số...).
    """
    if isinstance(raw_value, list):
        return raw_value

    if pd.isna(raw_value):
        return []

    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return parsed
        # Trường hợp chỉ có 1 câu, không phải list
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # Nếu parse thất bại, coi như cả chuỗi là 1 câu paraphrase duy nhất
        return [raw_value]


def load_humarin_pairs(raw_path: str = HUMARIN_RAW_PATH) -> pd.DataFrame:
    """
    Đọc file CSV gốc và trả về DataFrame đã "explode", với 2 cột:
        - original_text  : câu gốc
        - paraphrase_text: MỘT câu paraphrase tương ứng (1 dòng = 1 cặp)

    Nếu 1 câu gốc có 5 paraphrase, hàm này sẽ tạo ra 5 dòng riêng biệt,
    mỗi dòng đều có cùng original_text nhưng khác paraphrase_text.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dataset tại: {raw_path}\n"
            f"Hãy chạy download_dataset.py trước (trên môi trường có "
            f"kết nối internet tới HuggingFace, ví dụ Google Colab)."
        )

    df_raw = pd.read_csv(raw_path)

    required_cols = {"text", "paraphrases"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"File dataset thiếu cột bắt buộc: {missing}. "
            f"Các cột hiện có: {list(df_raw.columns)}. "
            f"Kiểm tra lại xem dataset tải về có đúng cấu trúc không."
        )

    df_raw["paraphrase_list"] = df_raw["paraphrases"].apply(_parse_paraphrase_list)

    # Explode: biến 1 dòng có list N câu thành N dòng riêng biệt
    df_exploded = df_raw.explode("paraphrase_list").reset_index(drop=True)

    df_pairs = pd.DataFrame({
        "original_text": df_exploded["text"],
        "paraphrase_text": df_exploded["paraphrase_list"],
    })

    # Loại bỏ dòng thiếu dữ liệu hoặc paraphrase rỗng/trùng với câu gốc
    df_pairs = df_pairs.dropna(subset=["original_text", "paraphrase_text"])
    df_pairs = df_pairs[
        df_pairs["original_text"].str.strip() != df_pairs["paraphrase_text"].str.strip()
    ]
    df_pairs = df_pairs.drop_duplicates().reset_index(drop=True)

    return df_pairs


def ensure_dirs():
    """Tạo sẵn các thư mục output nếu chưa tồn tại."""
    for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR]:
        os.makedirs(d, exist_ok=True)
