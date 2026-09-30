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

"""
convert_padben_format.py

-------------------------

Chuyển PADBen Task 1, 2, 3 từ dạng raw:

    idx
    sentence_pair
    label_pair

sang dạng single-sentence dùng cho Tầng 0:

    pair_id
    sentence_position
    sentence
    label
    source

Ý nghĩa:

    label = 0:
        Không thuộc nhóm AI-generated / AI-paraphrase
        theo định nghĩa của từng PADBen task.

    label = 1:
        Thuộc nhóm AI-generated / AI-paraphrase
        theo định nghĩa của từng PADBen task.

LƯU Ý QUAN TRỌNG
----------------

PADBen Task 1-3 có thể có dạng:

    sentence_pair = [sentence_A, sentence_B]
    label_pair    = [label_A, label_B]

Ví dụ:

    sentence_pair = [A, B]
    label_pair    = [0, 1]

có nghĩa:

    A -> label 0
    B -> label 1

KHÔNG có nghĩa:

    A + B -> label 1

Do đó file này sẽ flatten từng sentence thành một record
riêng cho Tầng 0.

Không sử dụng Task 4 và Task 5 trong file này.
"""

import argparse
import ast
import json
import os

import pandas as pd

from common import RAW_DIR, ensure_dirs


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "PADBen"

# Chỉ dùng Task 1, 2, 3 cho Tầng 0
SUPPORTED_TASKS = [1, 2, 3]


# ============================================================
# PATH
# ============================================================

PROCESSED_DIR = os.path.join(
    os.path.dirname(RAW_DIR),
    "processed",
    "padben"
)


# ============================================================
# HELPER
# ============================================================

def parse_list(value):
    """
    Chuyển một giá trị về Python list.

    Có thể xử lý:

        ['sentence A', 'sentence B']

    hoặc:

        ["sentence A", "sentence B"]

    hoặc trường hợp value đã là list.
    """

    if isinstance(value, list):
        return value

    if value is None:
        return None

    # pandas NaN
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    # --------------------------------------------------------
    # Thử JSON
    # --------------------------------------------------------

    try:
        result = json.loads(value)

        if isinstance(result, list):
            return result

    except Exception:
        pass

    # --------------------------------------------------------
    # Thử Python literal
    # --------------------------------------------------------

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except Exception:
        pass

    return None


# ============================================================
# LOAD
# ============================================================

def load_padben_json(input_path):
    """
    Đọc file JSON PADBen.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"\nKhông tìm thấy file:\n{input_path}\n\n"
            f"Hãy chạy download_padben.py trước."
        )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            "File PADBen không có dạng list records."
        )

    return pd.DataFrame(records)


# ============================================================
# INSPECT
# ============================================================

def inspect_task(task_id):
    """
    Kiểm tra cấu trúc thực tế của một task.
    """

    input_path = os.path.join(
        RAW_DIR,
        f"padben_sentence_pair_task{task_id}.json"
    )

    df = load_padben_json(input_path)

    print("\n" + "=" * 80)
    print(f"PADBen TASK {task_id} - INSPECTION")
    print("=" * 80)

    print(f"\nFile:")
    print(input_path)

    print(f"\nSố dòng: {len(df)}")

    print("\nCác cột:")
    print(list(df.columns))

    print("\n--- 3 dòng đầu tiên ---")

    for i in range(min(3, len(df))):

        print(f"\n[Dòng {i}]")

        for col in df.columns:

            value = df.iloc[i][col]

            print(
                f"{col}: {str(value)[:500]}"
            )

    # --------------------------------------------------------
    # Kiểm tra format sentence_pair / label_pair
    # --------------------------------------------------------

    if (
        "sentence_pair" in df.columns
        and "label_pair" in df.columns
    ):

        print(
            "\nPhát hiện format:"
            "\n  sentence_pair"
            "\n  label_pair"
        )

        sample_sentence = parse_list(
            df.iloc[0]["sentence_pair"]
        )

        sample_label = parse_list(
            df.iloc[0]["label_pair"]
        )

        print(
            "\nSample sentence_pair:"
        )
        print(sample_sentence)

        print(
            "\nSample label_pair:"
        )
        print(sample_label)

    else:

        print(
            "\nCẢNH BÁO:"
            "\nKhông tìm thấy đồng thời "
            "'sentence_pair' và 'label_pair'."
        )

    return df


# ============================================================
# CONVERT ONE TASK
# ============================================================

def convert_task(task_id):
    """
    Convert một PADBen task thành single-sentence dataset.

    Output:

        pair_id
        sentence_position
        sentence
        label
        source
    """

    if task_id not in SUPPORTED_TASKS:

        raise ValueError(
            f"Task {task_id} không được hỗ trợ.\n"
            f"Chỉ convert Task: {SUPPORTED_TASKS}"
        )

    input_path = os.path.join(
        RAW_DIR,
        f"padben_sentence_pair_task{task_id}.json"
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        f"single_task{task_id}.csv"
    )

    df_raw = load_padben_json(input_path)

    print("\n" + "=" * 80)
    print(f"CONVERT PADBen TASK {task_id}")
    print("=" * 80)

    # --------------------------------------------------------
    # Kiểm tra columns
    # --------------------------------------------------------

    required_columns = [
        "idx",
        "sentence_pair",
        "label_pair"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df_raw.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nTask {task_id} thiếu các cột:"
            f"\n{missing_columns}"
            f"\n\nCác cột hiện tại:"
            f"\n{list(df_raw.columns)}"
            f"\n\nHãy chạy:"
            f"\npython src/data_prep/convert_padben_format.py "
            f"--task {task_id} --inspect"
        )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    rows = []

    invalid_pairs = 0
    invalid_labels = 0
    empty_sentences = 0

    for _, row in df_raw.iterrows():

        pair_id = row["idx"]

        sentences = parse_list(
            row["sentence_pair"]
        )

        labels = parse_list(
            row["label_pair"]
        )

        # ----------------------------------------------------
        # Kiểm tra pair
        # ----------------------------------------------------

        if (
            sentences is None
            or labels is None
            or len(sentences) != 2
            or len(labels) != 2
        ):

            invalid_pairs += 1
            continue

        # ----------------------------------------------------
        # Tạo 2 records
        # ----------------------------------------------------

        for position in range(2):

            sentence = sentences[position]
            label = labels[position]

            # -----------------------------------------------
            # Kiểm tra sentence
            # -----------------------------------------------

            if (
                sentence is None
                or str(sentence).strip() == ""
            ):

                empty_sentences += 1
                continue

            # -----------------------------------------------
            # Kiểm tra label
            # -----------------------------------------------

            try:

                label = int(label)

            except (ValueError, TypeError):

                invalid_labels += 1
                continue

            if label not in [0, 1]:

                invalid_labels += 1
                continue

            # -----------------------------------------------
            # Append
            # -----------------------------------------------

            rows.append({

                "pair_id": pair_id,

                "sentence_position": position,

                "sentence": str(sentence).strip(),

                "label": label,

                "source": f"padben_task{task_id}"

            })

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows,
        columns=[
            "pair_id",
            "sentence_position",
            "sentence",
            "label",
            "source"
        ]
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "pair_id",
            "sentence_position"
        ]
    ).reset_index(drop=True)

    removed_duplicates = (
        before_duplicates - len(df)
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        f"\nSố pair raw:"
        f" {len(df_raw)}"
    )

    print(
        f"Số sentence hợp lệ:"
        f" {len(df)}"
    )

    print(
        f"Pair lỗi:"
        f" {invalid_pairs}"
    )

    print(
        f"Sentence rỗng:"
        f" {empty_sentences}"
    )

    print(
        f"Label lỗi:"
        f" {invalid_labels}"
    )

    print(
        f"Duplicate bị loại:"
        f" {removed_duplicates}"
    )

    if len(df) > 0:

        print(
            "\nPhân bố label:"
        )

        print(
            df["label"].value_counts(
                normalize=False
            ).sort_index()
        )

        print(
            "\nTỷ lệ label:"
        )

        print(
            df["label"].value_counts(
                normalize=True
            ).sort_index()
        )

        print(
            "\n5 dòng đầu:"
        )

        print(
            df.head().to_string(
                index=False
            )
        )

    print(
        f"\nĐã lưu:"
        f"\n{output_path}"
    )

    return df


# ============================================================
# MERGE TASK 1-3
# ============================================================

def merge_tasks():
    """
    Gộp single_task1.csv, single_task2.csv,
    single_task3.csv thành một dataset cho Tầng 0.

    Output:

        data/processed/padben/single_task1_2_3.csv
    """

    dataframes = []

    print("\n" + "=" * 80)
    print("MERGE PADBen TASK 1 + TASK 2 + TASK 3")
    print("=" * 80)

    for task_id in SUPPORTED_TASKS:

        path = os.path.join(
            PROCESSED_DIR,
            f"single_task{task_id}.csv"
        )

        if not os.path.exists(path):

            print(
                f"\nChưa có:"
                f"\n{path}"
            )

            print(
                f"\nHãy convert Task {task_id} trước."
            )

            continue

        df = pd.read_csv(path)

        dataframes.append(df)

        print(
            f"\nTask {task_id}:"
            f" {len(df)} sentences"
        )

    if not dataframes:

        raise ValueError(
            "Không tìm thấy dữ liệu Task 1-3 "
            "để merge."
        )

    merged = pd.concat(
        dataframes,
        ignore_index=True
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        "single_task1_2_3.csv"
    )

    merged.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n" + "-" * 80
    )

    print(
        f"Tổng số sentence:"
        f" {len(merged)}"
    )

    print(
        "\nPhân bố source:"
    )

    print(
        merged["source"].value_counts()
    )

    print(
        "\nPhân bố label:"
    )

    print(
        merged["label"].value_counts()
    )

    print(
        "\nĐã lưu:"
        f"\n{output_path}"
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert PADBen Task 1-3 "
            "cho Tầng 0"
        )
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        help="Task cần xử lý: 1, 2 hoặc 3"
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Chỉ inspect task, không convert"
    )

    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert task thành single-sentence"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge Task 1 + 2 + 3"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert cả Task 1, 2, 3 rồi merge"
    )

    args = parser.parse_args()

    ensure_dirs()

    # --------------------------------------------------------
    # INSPECT
    # --------------------------------------------------------

    if args.inspect:

        if args.task is None:

            raise ValueError(
                "--inspect cần --task"
            )

        inspect_task(
            args.task
        )

        return

    # --------------------------------------------------------
    # CONVERT ONE TASK
    # --------------------------------------------------------

    if args.convert:

        if args.task is None:

            raise ValueError(
                "--convert cần --task"
            )

        convert_task(
            args.task
        )

        return

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    if args.merge:

        merge_tasks()

        return

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    if args.all:

        for task_id in SUPPORTED_TASKS:

            convert_task(
                task_id
            )

        merge_tasks()

        return

    # --------------------------------------------------------
    # NO ARGUMENT
    # --------------------------------------------------------

    parser.print_help()


if __name__ == "__main__":
    main()