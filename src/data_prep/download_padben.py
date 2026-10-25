"""
download_padben.py
-------------------
Tải dataset phụ PADBen (JonathanZha/PADBen) từ HuggingFace.

VAI TRÒ: prepare_datasets.py dùng Task 1/2/3 cho bước 0 và giữ các nhóm
test của Task 4/5 để đánh giá. Nhãn task phải được ánh xạ trước khi gộp.

LƯU Ý QUAN TRỌNG:
    - Cần kết nối internet tới HuggingFace Hub, chạy trên Colab/Kaggle.
    - Trước khi dùng thật, hãy IN THỬ vài dòng đầu và kiểm tra tên cột
      bằng mắt (xem hàm main() bên dưới) — vì PADBen có nhiều "task"
      khác nhau (task1 đến task5), tên cột có thể thay đổi theo config.

Cách chạy:
    python src/data_prep/download_padben.py
"""

import os
from datasets import load_dataset, get_dataset_config_names

from common import RAW_DIR, ensure_dirs

DEFAULT_OUTPUT_PATH = os.path.join(RAW_DIR, "padben_sentence_pair_task1.json")


def list_available_configs(dataset_name: str = "JonathanZha/PADBen") -> list:
    """In ra danh sách các config (task) có sẵn để bạn chọn đúng cái cần."""
    configs = get_dataset_config_names(dataset_name)
    print(f"Các config có sẵn trong '{dataset_name}': {configs}")
    return configs


def download_padben(
    config_name: str = "sentence-pair-task1",
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    ensure_dirs()

    print(f"Đang tải PADBen, config = '{config_name}' ...")
    dataset = load_dataset("JonathanZha/PADBen", config_name)

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    print(f"Đã tải xong: {len(df)} dòng, split = '{split_name}'")
    print(f"Các cột hiện có: {list(df.columns)}")
    print("--- 3 dòng đầu tiên (kiểm tra bằng mắt trước khi dùng) ---")
    print(df.head(3))

    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"Đã lưu vào: {output_path}")
    return df


if __name__ == "__main__":
    # Bước 1: xem có những config nào, chọn đúng cái gần với bài toán
    # "sentence-pair" / "task1" nhất trước khi tải chính thức.
    # list_available_configs()
    # download_padben()
    
    for task in range(1, 6):
        config_name = f"sentence-pair-task{task}"
        output_path = os.path.join(
            RAW_DIR,
            f"padben_sentence_pair_task{task}.json"
        )

        print(f"\n{'=' * 60}")
        print(f"ĐANG TẢI {config_name}")
        print(f"{'=' * 60}")

        download_padben(
            config_name=config_name,
            output_path=output_path
        )
