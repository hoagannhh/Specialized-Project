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

import os

import pandas as pd

from common import load_humarin_pairs, PROCESSED_DIR, ensure_dirs

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "single_labeled.csv")
RANDOM_SEED = 42


def build_single_label_dataset() -> pd.DataFrame:
    ensure_dirs()

    df_pairs = load_humarin_pairs()
    print(f"Đã load {len(df_pairs)} cặp gốc để xây dữ liệu Bước 0")

    # Lớp 0 — người viết: loại trùng lặp vì 1 original_text
    # có thể xuất hiện nhiều lần (ứng với nhiều paraphrase khác nhau)
    df_human = (
        df_pairs[["original_text"]]
        .drop_duplicates()
        .rename(columns={"original_text": "sentence"})
    )
    df_human["label"] = 0

    # Lớp 1 — đã qua AI: mỗi paraphrase là 1 mẫu riêng (không cần
    # loại trùng vì về mặt lý thuyết mỗi paraphrase là 1 câu khác nhau,
    # nhưng vẫn loại phòng trường hợp dataset có lặp y hệt)
    df_ai = (
        df_pairs[["paraphrase_text"]]
        .drop_duplicates()
        .rename(columns={"paraphrase_text": "sentence"})
    )
    df_ai["label"] = 1

    df_final = pd.concat([df_human, df_ai], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n_human = (df_final["label"] == 0).sum()
    n_ai = (df_final["label"] == 1).sum()
    print(f"Tổng số mẫu: {len(df_final)} (label=0 người viết: {n_human}, "
          f"label=1 AI-touched: {n_ai})")

    if abs(n_human - n_ai) / max(n_human, n_ai) > 0.3:
        print(
            "CẢNH BÁO: 2 lớp đang khá mất cân bằng "
            f"(chênh lệch > 30%). Có thể cần thêm bước cân bằng dữ liệu "
            f"(oversampling lớp thiểu số hoặc class_weight khi train)."
        )

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Đã lưu vào: {OUTPUT_PATH}")

    return df_final


if __name__ == "__main__":
    build_single_label_dataset()
