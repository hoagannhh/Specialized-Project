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
import os

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers")

from model import BiEncoder, TripletLoss, load_tokenizer, DEFAULT_PRETRAINED_NAME


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
def evaluate(model: BiEncoder, dataloader, device) -> float:
    """
    Đánh giá bằng % triplet mà model xếp ĐÚNG THỨ TỰ
    (similarity(A,P) > similarity(A,N)) — độ đo chuẩn cho bài toán
    Triplet, tương đương TripletEvaluator của sentence-transformers.
    """
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        anchor_emb = model(batch["anchor_ids"].to(device), batch["anchor_mask"].to(device))
        positive_emb = model(batch["positive_ids"].to(device), batch["positive_mask"].to(device))
        negative_emb = model(batch["negative_ids"].to(device), batch["negative_mask"].to(device))

        sim_ap = torch.sum(anchor_emb * positive_emb, dim=1)
        sim_an = torch.sum(anchor_emb * negative_emb, dim=1)

        correct += (sim_ap > sim_an).sum().item()
        total += sim_ap.size(0)

    return correct / total if total > 0 else 0.0


# ============================================================
# TRAIN
# ============================================================

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
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train: {len(df_train):,} triplet | Val: {len(df_val):,} triplet")

    tokenizer = load_tokenizer(pretrained_name)

    train_dataset = TripletDataset(df_train, tokenizer, max_length)
    val_dataset = TripletDataset(df_val, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = BiEncoder(pretrained_name=pretrained_name).to(device)
    loss_fn = TripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            anchor_emb = model(batch["anchor_ids"].to(device), batch["anchor_mask"].to(device))
            positive_emb = model(batch["positive_ids"].to(device), batch["positive_mask"].to(device))
            negative_emb = model(batch["negative_ids"].to(device), batch["negative_mask"].to(device))

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader, device)

        print(f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_accuracy(AP>AN)={val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(output_dir, "tang3_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Lưu checkpoint tốt nhất: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> Early stopping (không cải thiện sau {patience} epoch)")
                break

    print(f"\n=== Hoàn tất. Val accuracy tốt nhất: {best_val_accuracy:.4f} ===")

    # Lưu tokenizer cùng checkpoint để inference.py dùng lại đúng
    tokenizer.save_pretrained(output_dir)

    return model


if __name__ == "__main__":
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
    args = parser.parse_args()

    train(
        train_csv=args.train, val_csv=args.val, output_dir=args.output_dir,
        pretrained_name=args.pretrained_name, max_length=args.max_length,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        margin=args.margin, num_epochs=args.num_epochs, patience=args.patience,
    )
