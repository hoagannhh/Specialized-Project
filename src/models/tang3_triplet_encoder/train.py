"""
Tầng 3 — Training loop (tự viết bằng PyTorch thuần)
=======================================================

Cách chạy:
    python src/models/tang3_triplet_encoder/train.py \\
        --train data/splits/triplet_data_train.csv \\
        --val data/splits/triplet_data_val.csv \\
        --output-dir checkpoints/tang3
"""

import argparse
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.runtime import seed_everything, save_bundle, restore_best, select_threshold
import os
from src.models.training_checkpoint import TrainingRun, pause_cleanly, add_checkpoint_arguments

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers")

from src.models.tang3_triplet_encoder.model import BiEncoder, TripletLoss, load_tokenizer, DEFAULT_PRETRAINED_NAME


# ============================================================
# DATASET
# ============================================================

class TripletDataset(Dataset):
    """
    Mỗi mẫu trả về 3 cặp (input_ids, attention_mask) đã token hóa —
    1 cho Anchor, 1 cho Positive, 1 cho Negative.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.anchors = df["anchor"].astype(str).tolist()
        self.positives = df["positive"].astype(str).tolist()
        self.negatives = df["negative"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.anchors)

    def _tokenize(self, text):
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        a_ids, a_mask = self._tokenize(self.anchors[idx])
        p_ids, p_mask = self._tokenize(self.positives[idx])
        n_ids, n_mask = self._tokenize(self.negatives[idx])

        return {
            "anchor_ids": a_ids, "anchor_mask": a_mask,
            "positive_ids": p_ids, "positive_mask": p_mask,
            "negative_ids": n_ids, "negative_mask": n_mask,
        }


# ============================================================
# ĐÁNH GIÁ
# ============================================================

@torch.no_grad()
def evaluate(model: BiEncoder, dataloader, device, check_time=None, loss_fn=None):
    """
    Đánh giá bằng % triplet mà model xếp ĐÚNG THỨ TỰ
    (similarity(A,P) > similarity(A,N)) — độ đo chuẩn cho bài toán
    Triplet, tương đương TripletEvaluator của sentence-transformers.
    """
    model.eval()
    correct = 0
    losses = []
    total = 0

    for batch in dataloader:
        if check_time is not None:
            check_time()
        anchor_emb = model(batch["anchor_ids"].to(device), batch["anchor_mask"].to(device))
        positive_emb = model(batch["positive_ids"].to(device), batch["positive_mask"].to(device))
        negative_emb = model(batch["negative_ids"].to(device), batch["negative_mask"].to(device))

        if loss_fn is not None:
            losses.append(loss_fn(anchor_emb, positive_emb, negative_emb).item())
        sim_ap = torch.sum(anchor_emb * positive_emb, dim=1)
        sim_an = torch.sum(anchor_emb * negative_emb, dim=1)

        correct += (sim_ap > sim_an).sum().item()
        total += sim_ap.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return (accuracy, float(np.mean(losses))) if loss_fn is not None else accuracy


# ============================================================
# TRAIN
# ============================================================

@pause_cleanly
def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    margin: float = 0.5,
    num_epochs: int = 3,
    patience: int = 2,
    seed: int = 42,
    resume_from: str = None,
    save_every_steps: int = 500,
    save_every_minutes: float = 10,
    max_runtime_minutes: float = None,
):
    if min(batch_size, max_length, num_epochs, patience) < 1:
        raise ValueError('batch_size, max_length, num_epochs and patience must be positive')
    run = TrainingRun(output_dir, 3, train_csv, val_csv,
                      dict(pretrained_name=str(pretrained_name), max_length=max_length, batch_size=batch_size,
                  learning_rate=learning_rate, patience=patience, seed=seed, margin=margin),
                      resume_from, save_every_steps, save_every_minutes, max_runtime_minutes)
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    if df_train.empty or df_val.empty:
        raise ValueError('Training and validation datasets must not be empty')
    print(f"Train: {len(df_train):,} triplet | Val: {len(df_val):,} triplet")

    tokenizer = load_tokenizer(run.tokenizer_source)

    train_dataset = TripletDataset(df_train, tokenizer, max_length)
    val_dataset = TripletDataset(df_val, tokenizer, max_length)

    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                            generator=torch.Generator().manual_seed(seed))

    model = BiEncoder(pretrained_name=pretrained_name, encoder_config=run.encoder_config).to(device)
    loss_fn = TripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    run.attach(model, tokenizer, optimizer)
    best_val_accuracy = run.state['best_metric']
    epochs_without_improvement = run.state['bad_epochs']

    if num_epochs < run.state['epoch'] and not run.state['finished']:
        raise ValueError('num_epochs là tổng số epoch, phải đủ để tiếp tục checkpoint hiện tại')
    for epoch in range(run.state['epoch'], num_epochs + 1):
        if run.state['finished']:
            break
        model.train()
        train_loader = run.loader(train_dataset, batch_size)

        for batch_index, batch in enumerate(train_loader, start=run.state['next_batch']):
            run.check_time()
            anchor_emb = model(batch["anchor_ids"].to(device), batch["anchor_mask"].to(device))
            positive_emb = model(batch["positive_ids"].to(device), batch["positive_mask"].to(device))
            negative_emb = model(batch["negative_ids"].to(device), batch["negative_mask"].to(device))

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run.after_step(batch_index, loss.item())

        # Lưu trước validation: bị ngắt ở đây chỉ cần chạy lại validation.
        run.save()
        avg_train_loss = run.state['total_loss'] / run.state['next_batch']
        val_accuracy, val_loss = evaluate(model, val_loader, device, check_time=run.check_time, loss_fn=loss_fn)

        print(f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_accuracy(AP>AN)={val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(output_dir, "tang3_best.pt")
            save_bundle(model, tokenizer, output_dir, 3, pretrained_name, max_length, seed=seed, margin=margin, epoch=epoch, val_accuracy=float(val_accuracy))
            print(f"  -> Lưu checkpoint tốt nhất: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
        stopped = epochs_without_improvement >= patience
        run.finish_epoch(best_val_accuracy, epochs_without_improvement, stopped or epoch == num_epochs,
                         metrics=dict(train_loss=avg_train_loss, val_loss=val_loss,
                                      val_accuracy=float(val_accuracy)))
        if stopped:
            print(f"  -> Early stopping (không cải thiện sau {patience} epoch)")
            break

    print(f"\n=== Hoàn tất. Val accuracy tốt nhất: {best_val_accuracy:.4f} ===")

    # Lưu tokenizer cùng checkpoint để inference.py dùng lại đúng
    tokenizer.save_pretrained(output_dir)

    restore_best(model, output_dir, 3)
    return model


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", default="checkpoints/tang3")
    parser.add_argument("--pretrained-name", default=DEFAULT_PRETRAINED_NAME)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    add_checkpoint_arguments(parser)
    args = parser.parse_args()

    train(
        train_csv=args.train, val_csv=args.val, output_dir=args.output_dir,
        pretrained_name=args.pretrained_name, max_length=args.max_length,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        margin=args.margin, num_epochs=args.num_epochs, patience=args.patience, seed=args.seed,
        resume_from=args.resume_from, save_every_steps=args.save_every_steps,
        save_every_minutes=args.save_every_minutes, max_runtime_minutes=args.max_runtime_minutes,
    )
