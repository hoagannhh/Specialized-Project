"""
Tầng 3 — Huấn luyện Bi-Encoder bằng Triplet Loss
===================================================

Dùng trực tiếp thư viện `sentence-transformers` — đã có sẵn
TripletLoss, InputExample, SentenceTransformer.fit(), không cần tự
viết training loop từ đầu như Tầng 0 (vì Tầng 0 cần custom nhiều
hơn do có class_weight đặc thù).

Kiến trúc: 1 ENCODER DUY NHẤT dùng CHUNG trọng số cho cả 3 nhánh
(Anchor, Positive, Negative) — khác với TriPlaNet (2025) dùng 3
encoder riêng tham số. Đây là phiên bản Triplet Network kinh điển,
đơn giản hóa cho phù hợp quy mô đồ án (xem giải thích chi tiết đã
thống nhất ở phần thiết kế kiến trúc).

Cách chạy:
    python src/models/tang3_triplet_encoder/train.py \\
        --train data/splits/triplet_data_train.csv \\
        --val data/splits/triplet_data_val.csv \\
        --output-dir checkpoints/tang3
"""

import argparse
import os

import pandas as pd

try:
    from sentence_transformers import InputExample, losses
    from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError(
        f"Thiếu thư viện: {e}. Chạy: pip install sentence-transformers torch"
    )

from model import build_triplet_encoder, DEFAULT_BASE_MODEL


def load_triplet_examples(csv_path: str) -> list:
    """
    Đọc file triplet_data.csv (anchor, positive, negative) và chuyển
    thành danh sách InputExample — định dạng mà sentence-transformers
    yêu cầu.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"anchor", "positive", "negative"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"File thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}")

    examples = [
        InputExample(texts=[str(row["anchor"]), str(row["positive"]), str(row["negative"])])
        for _, row in df.iterrows()
    ]
    print(f"Đã load {len(examples):,} triplet examples từ {csv_path}")
    return examples


def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    batch_size: int = 16,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    triplet_margin: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    model = build_triplet_encoder(base_model)

    train_examples = load_triplet_examples(train_csv)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # TripletLoss: kéo gần (anchor, positive), đẩy xa (anchor, negative)
    # với khoảng cách margin tối thiểu — đúng công thức đã trình bày
    # trong Chương 3: L = max(0, d(A,P) - d(A,N) + margin)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=triplet_margin,
    )

    # Evaluator: đo % triplet mà model xếp đúng thứ tự
    # (d(A,P) < d(A,N)) trên tập validation sau mỗi epoch
    df_val = pd.read_csv(val_csv)
    val_evaluator = TripletEvaluator(
        anchors=df_val["anchor"].astype(str).tolist(),
        positives=df_val["positive"].astype(str).tolist(),
        negatives=df_val["negative"].astype(str).tolist(),
        name="tang3_val",
    )

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"Bắt đầu huấn luyện: {num_epochs} epoch, {len(train_dataloader)} bước/epoch, "
          f"warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"\nHoàn tất huấn luyện. Model tốt nhất đã lưu tại: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", default="checkpoints/tang3")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--triplet-margin", type=float, default=0.5)
    args = parser.parse_args()

    train(
        train_csv=args.train,
        val_csv=args.val,
        output_dir=args.output_dir,
        base_model=args.base_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        triplet_margin=args.triplet_margin,
    )
