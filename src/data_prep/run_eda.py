"""
run_eda.py
----------
Kiểm tra chất lượng dữ liệu của cả 3 tập (Tầng 0, Tầng 3, Tầng 4)
TRƯỚC KHI bỏ thời gian GPU ra huấn luyện. Script in ra số liệu thống
kê quan trọng và lưu biểu đồ (.png) vào thư mục report.

Mỗi hàm check_* tương ứng với 1 mục trong checklist đã thống nhất:
    - Tầng 0: phân bố độ dài theo label, phân bố nguồn theo split,
              trùng lặp nội dung giữa các split
    - Tầng 3: phân bố similarity (Anchor-Positive) vs (Anchor-Negative),
              phân bố độ dài, in mẫu ngẫu nhiên để tự kiểm tra bằng mắt
    - Tầng 4: xác nhận cân bằng nhãn, phân bố độ dài 2 vế, thống kê
              các tập cross-eval (PADBen Task 4, 5)

Cách chạy (ví dụ):
    python src/data_prep/run_eda.py --tang0-train data/splits/tang0_train.csv --tang0-val data/splits/tang0_val.csv --tang0-test data/splits/tang0_test.csv \\
        --triplet data/splits/triplet_data_train.csv \\
        --pairwise data/splits/pairwise_labeled_train.csv \\
        --output-dir evaluation/eda_results
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # không cần hiển thị màn hình, chỉ lưu file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option("display.width", 120)

SIMILARITY_SAMPLE_SIZE = 50_000  # Lấy mẫu để tính similarity cho nhanh, không cần full data


def _save_and_close(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Đã lưu biểu đồ: {path}")


def _print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------
# TẦNG 0 — dữ liệu đơn-câu (merged humarin + PADBen task 1,2,3)
# ---------------------------------------------------------------------

def check_tang0(train_path: str, val_path: str, test_path: str, output_dir: str):
    _print_section("TẦNG 0 — Kiểm tra dữ liệu đơn-câu (train/val/test)")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\n--- Tập {name}: {len(df):,} dòng ---")
        print("Tỷ lệ nhãn:")
        print(df["label"].value_counts(normalize=True).rename("tỷ lệ").to_string())
        if "source" in df.columns:
            print("Tỷ lệ nguồn (source):")
            print(df["source"].value_counts(normalize=True).rename("tỷ lệ").to_string())

    # 1. Phân bố độ dài câu theo label (trên tập train)
    df_train = df_train.copy()
    df_train["n_words"] = df_train["sentence"].astype(str).str.split().str.len()

    print("\n--- Độ dài câu (số từ) theo label, tập train ---")
    print(df_train.groupby("label")["n_words"].describe().to_string())

    mean_diff = (
        df_train.loc[df_train["label"] == 1, "n_words"].mean()
        - df_train.loc[df_train["label"] == 0, "n_words"].mean()
    )
    print(f"\nChênh lệch độ dài trung bình (label 1 - label 0): {mean_diff:+.2f} từ")
    if abs(mean_diff) > 3:
        print(
            "  CẢNH BÁO: chênh lệch độ dài giữa 2 lớp khá lớn (>3 từ). "
            "Model có nguy cơ học 'mẹo' đếm độ dài thay vì học phong cách viết "
            "thật. Cân nhắc kiểm tra kỹ hoặc chuẩn hóa độ dài khi sampling."
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    for label_value, color, name in [(0, "tab:blue", "Người viết (0)"),
                                      (1, "tab:orange", "AI-touched (1)")]:
        subset = df_train.loc[df_train["label"] == label_value, "n_words"]
        ax.hist(subset, bins=40, alpha=0.5, density=True, label=name, color=color)
    ax.set_xlabel("Số từ trong câu")
    ax.set_ylabel("Mật độ")
    ax.set_title("Tầng 0 — Phân bố độ dài câu theo nhãn (tập train)")
    ax.legend()
    _save_and_close(fig, output_dir, "tang0_do_dai_theo_nhan.png")

    # 2. Kiểm tra trùng lặp NỘI DUNG (không chỉ group_id) giữa các split
    print("\n--- Kiểm tra trùng lặp nội dung câu giữa các tập ---")
    norm = lambda s: s.astype(str).str.strip().str.lower()
    set_train = set(norm(df_train["sentence"]))
    set_val = set(norm(df_val["sentence"]))
    set_test = set(norm(df_test["sentence"]))

    overlap_train_val = set_train & set_val
    overlap_train_test = set_train & set_test
    overlap_val_test = set_val & set_test

    print(f"  Trùng train-val : {len(overlap_train_val):,} câu")
    print(f"  Trùng train-test: {len(overlap_train_test):,} câu")
    print(f"  Trùng val-test  : {len(overlap_val_test):,} câu")

    total_overlap = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
    if total_overlap > 0:
        print(
            "  CẢNH BÁO: có câu trùng nội dung giữa các tập dù khác group_id "
            "(thường do câu ngắn/generic lặp lại tự nhiên trong corpus). "
            "Nếu số lượng này đáng kể so với tổng số dòng, cân nhắc lọc bớt."
        )
    else:
        print("  Không phát hiện trùng lặp nội dung giữa các tập.")


# ---------------------------------------------------------------------
# TẦNG 3 — dữ liệu Triplet (anchor, positive, negative)
# ---------------------------------------------------------------------

def _paired_cosine_similarity(vectorizer: TfidfVectorizer, text_a: list, text_b: list) -> np.ndarray:
    """
    Tính cosine similarity TỪNG CẶP (row-wise), KHÔNG tính toàn bộ ma
    trận N x N -> an toàn bộ nhớ ngay cả với N lớn, vì chỉ cần N phép
    tính, không phải N^2.
    """
    vec_a = vectorizer.transform(text_a)
    vec_b = vectorizer.transform(text_b)
    # TfidfVectorizer mặc định đã chuẩn hóa L2 (norm='l2'), nên dot
    # product từng cặp hàng chính là cosine similarity
    product = vec_a.multiply(vec_b)
    sims = np.asarray(product.sum(axis=1)).flatten()
    return sims


def check_triplet(triplet_path: str, output_dir: str, sample_size: int = SIMILARITY_SAMPLE_SIZE):
    _print_section("TẦNG 3 — Kiểm tra dữ liệu Triplet (Anchor, Positive, Negative)")

    df = pd.read_csv(triplet_path)
    print(f"Tổng số dòng: {len(df):,}")

    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Lấy mẫu {sample_size:,} dòng để tính similarity cho nhanh")
    else:
        df_sample = df

    # 1. Phân bố độ dài
    for col in ["anchor", "positive", "negative"]:
        lengths = df_sample[col].astype(str).str.split().str.len()
        print(f"  Độ dài '{col}': trung bình={lengths.mean():.1f} từ, "
              f"min={lengths.min()}, max={lengths.max()}")

    # 2. Similarity Anchor-Positive vs Anchor-Negative
    vectorizer = TfidfVectorizer()
    all_texts = (
        df_sample["anchor"].astype(str).tolist()
        + df_sample["positive"].astype(str).tolist()
        + df_sample["negative"].astype(str).tolist()
    )
    vectorizer.fit(all_texts)

    sim_ap = _paired_cosine_similarity(vectorizer, df_sample["anchor"], df_sample["positive"])
    sim_an = _paired_cosine_similarity(vectorizer, df_sample["anchor"], df_sample["negative"])

    print(f"\n  Similarity (Anchor, Positive): trung bình={sim_ap.mean():.3f}, "
          f"trung vị={np.median(sim_ap):.3f}")
    print(f"  Similarity (Anchor, Negative): trung bình={sim_an.mean():.3f}, "
          f"trung vị={np.median(sim_an):.3f}")

    gap = sim_ap.mean() - sim_an.mean()
    print(f"  Khoảng cách trung bình (AP - AN): {gap:+.3f}")
    if gap < 0.1:
        print(
            "  CẢNH BÁO: khoảng cách giữa Positive và Negative khá nhỏ (<0.1). "
            "Dữ liệu có thể 'khó học' quá mức hoặc hard-negative đang chọn "
            "những câu quá giống Positive. Nên lấy mẫu kiểm tra bằng mắt."
        )

    n_overlap = (sim_an >= sim_ap).sum()
    pct_overlap = n_overlap / len(df_sample) * 100
    print(f"  Số dòng mà Negative >= Positive về similarity: {n_overlap:,} "
          f"({pct_overlap:.1f}%)")
    if pct_overlap > 15:
        print(
            "  CẢNH BÁO: tỷ lệ Negative 'giống Anchor hơn cả Positive' khá cao "
            "(>15%). Đây là các triplet 'khó' bất thường, có thể do nhiễu "
            "dữ liệu hoặc hard-negative mining quá gắt. Cân nhắc giảm top_k_pool."
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sim_ap, bins=40, alpha=0.5, density=True, label="Anchor–Positive", color="tab:green")
    ax.hist(sim_an, bins=40, alpha=0.5, density=True, label="Anchor–Negative", color="tab:red")
    ax.set_xlabel("Cosine similarity (TF-IDF)")
    ax.set_ylabel("Mật độ")
    ax.set_title("Tầng 3 — Phân bố similarity: Positive vs Negative")
    ax.legend()
    _save_and_close(fig, output_dir, "tang3_similarity_positive_vs_negative.png")

    # 3. In vài dòng mẫu ngẫu nhiên để tự kiểm tra bằng mắt
    print("\n--- 5 dòng mẫu ngẫu nhiên để kiểm tra bằng mắt ---")
    for _, row in df_sample.sample(n=min(5, len(df_sample)), random_state=1).iterrows():
        print(f"\n  Anchor  : {row['anchor']}")
        print(f"  Positive: {row['positive']}")
        print(f"  Negative: {row['negative']}")


# ---------------------------------------------------------------------
# TẦNG 4 — dữ liệu Cặp-có-nhãn (sentence_1, sentence_2, label)
# ---------------------------------------------------------------------

def check_pairwise(pairwise_path: str, output_dir: str,
                    cross_eval_paths: dict = None):
    _print_section("TẦNG 4 — Kiểm tra dữ liệu Cặp-có-nhãn")

    df = pd.read_csv(pairwise_path)
    print(f"Tổng số dòng: {len(df):,}")

    print("\nTỷ lệ nhãn:")
    label_ratio = df["label"].value_counts(normalize=True)
    print(label_ratio.rename("tỷ lệ").to_string())
    if abs(label_ratio.get(0, 0) - label_ratio.get(1, 0)) > 0.1:
        print("  CẢNH BÁO: lệch khỏi 50/50 hơn 10 điểm %, kiểm tra lại bước build_pairwise.")
    else:
        print("  Cân bằng tốt (~50/50).")

    len_1 = df["sentence_1"].astype(str).str.split().str.len()
    len_2 = df["sentence_2"].astype(str).str.split().str.len()
    print(f"\n  Độ dài sentence_1: trung bình={len_1.mean():.1f} từ")
    print(f"  Độ dài sentence_2: trung bình={len_2.mean():.1f} từ")
    print(f"  Chênh lệch trung bình: {(len_2.mean() - len_1.mean()):+.1f} từ")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(len_1, bins=40, alpha=0.5, density=True, label="sentence_1", color="tab:blue")
    ax.hist(len_2, bins=40, alpha=0.5, density=True, label="sentence_2", color="tab:orange")
    ax.set_xlabel("Số từ")
    ax.set_ylabel("Mật độ")
    ax.set_title("Tầng 4 — Phân bố độ dài 2 vế câu")
    ax.legend()
    _save_and_close(fig, output_dir, "tang4_do_dai_2_ve.png")

    # Thống kê các tập cross-eval (PADBen Task 4, 5) nếu có truyền vào
    if cross_eval_paths:
        print("\n--- Thống kê các tập cross-eval (PADBen) ---")
        for name, path in cross_eval_paths.items():
            if not os.path.exists(path):
                print(f"  [{name}] KHÔNG tìm thấy file: {path}")
                continue
            df_ce = pd.read_csv(path)
            print(f"\n  [{name}] {len(df_ce):,} dòng, các cột: {list(df_ce.columns)}")
            if len(df_ce) < 200:
                print(
                    f"    CẢNH BÁO: chỉ có {len(df_ce)} mẫu — khá ít để kết luận "
                    f"có ý nghĩa thống kê. Kết quả trên tập này nên coi là "
                    f"tham khảo, không nên khái quát hóa quá mạnh."
                )
            if "label" in df_ce.columns:
                print("    Tỷ lệ nhãn:")
                print("    " + df_ce["label"].value_counts(normalize=True).to_string().replace("\n", "\n    "))
            if "paraphrase_round" in df_ce.columns:
                print("    Phân bố paraphrase_round:")
                print("    " + df_ce["paraphrase_round"].value_counts().to_string().replace("\n", "\n    "))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tang0-train", default=None)
    parser.add_argument("--tang0-val", default=None)
    parser.add_argument("--tang0-test", default=None)
    parser.add_argument("--triplet", default=None, help="Đường dẫn tới triplet_data (train hoặc full)")
    parser.add_argument("--pairwise", default=None, help="Đường dẫn tới pairwise_labeled (train hoặc full)")
    parser.add_argument("--padben-task4", default=None)
    parser.add_argument("--padben-task5", default=None)
    parser.add_argument("--output-dir", default="evaluation/eda_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tang0_train and args.tang0_val and args.tang0_test:
        check_tang0(args.tang0_train, args.tang0_val, args.tang0_test, args.output_dir)
    else:
        print("Bỏ qua kiểm tra Tầng 0 (thiếu --tang0-train/--tang0-val/--tang0-test)")

    if args.triplet:
        check_triplet(args.triplet, args.output_dir)
    else:
        print("Bỏ qua kiểm tra Tầng 3 (thiếu --triplet)")

    if args.pairwise:
        cross_eval_paths = {}
        if args.padben_task4:
            cross_eval_paths["PADBen Task 4"] = args.padben_task4
        if args.padben_task5:
            cross_eval_paths["PADBen Task 5"] = args.padben_task5
        check_pairwise(args.pairwise, args.output_dir, cross_eval_paths or None)
    else:
        print("Bỏ qua kiểm tra Tầng 4 (thiếu --pairwise)")

    print(f"\nHoàn tất EDA. Biểu đồ đã lưu trong: {args.output_dir}")


if __name__ == "__main__":
    main()
