"""
Tầng 4 — Huấn luyện Cross-Encoder
===================================

Dùng class CrossEncoder có sẵn trong sentence-transformers — đã
tích hợp sẵn kiến trúc [CLS] A [SEP] B [SEP] + classifier head +
Binary Cross-Entropy Loss, không cần tự viết từ đầu.

Cách chạy:
    python src/models/tang4_cross_encoder/train.py \\
        --train data/splits/pairwise_labeled_train.csv \\
        --val data/splits/pairwise_labeled_val.csv \\
        --output-dir checkpoints/tang4
"""

import argparse
import os

import pandas as pd

try:
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError(
        f"Thiếu thư viện: {e}. Chạy: pip install sentence-transformers torch"
    )

from model import build_cross_encoder, DEFAULT_BASE_MODEL


def load_pairwise_examples(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    required_cols = {"sentence_1", "sentence_2", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"File thiếu cột: {missing}. Các cột hiện có: {list(df.columns)}")

    examples = [
        InputExample(texts=[str(row["sentence_1"]), str(row["sentence_2"])], label=int(row["label"]))
        for _, row in df.iterrows()
    ]
    print(f"Đã load {len(examples):,} pairwise examples từ {csv_path}")
    return examples


def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    batch_size: int = 16,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    max_length: int = 256,
):
    os.makedirs(output_dir, exist_ok=True)

    model = build_cross_encoder(base_model, max_length=max_length)

    train_examples = load_pairwise_examples(train_csv)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    df_val = pd.read_csv(val_csv)
    val_evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=list(zip(df_val["sentence_1"].astype(str), df_val["sentence_2"].astype(str))),
        labels=df_val["label"].astype(int).tolist(),
        name="tang4_val",
    )

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(f"Bắt đầu huấn luyện: {num_epochs} epoch, {len(train_dataloader)} bước/epoch, "
          f"warmup_steps={warmup_steps}")

    model.fit(
        train_dataloader=train_dataloader,
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
    parser.add_argument("--output-dir", default="checkpoints/tang4")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    train(
        train_csv=args.train,
        val_csv=args.val,
        output_dir=args.output_dir,
        base_model=args.base_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
    )
