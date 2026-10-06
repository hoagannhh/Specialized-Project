"""
Tầng 4 — Training loop (tự viết bằng PyTorch thuần)
=======================================================

Cách chạy:
    python src/models/tang4_cross_encoder/train.py \\
        --train data/splits/pairwise_labeled_train.csv \\
        --val data/splits/pairwise_labeled_val.csv \\
        --output-dir checkpoints/tang4
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import f1_score, classification_report
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install torch transformers scikit-learn")

from model import CrossEncoderModel, load_tokenizer, DEFAULT_PRETRAINED_NAME


# ============================================================
# DATASET
# ============================================================

class PairwiseDataset(Dataset):
    """
    Mỗi mẫu: ghép sentence_1 + sentence_2 thành 1 chuỗi [CLS] s1
    [SEP] s2 [SEP] (tokenizer tự xử lý khi truyền 2 chuỗi cùng lúc).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.sentence_1 = df["sentence_1"].astype(str).tolist()
        self.sentence_2 = df["sentence_2"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()  # float vì BCEWithLogitsLoss cần float
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentence_1[idx], self.sentence_2[idx],  # truyền 2 câu -> tokenizer tự ghép [CLS] s1 [SEP] s2 [SEP]
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


# ============================================================
# ĐÁNH GIÁ
# ============================================================

@torch.no_grad()
def evaluate(model: CrossEncoderModel, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        probs = model.predict_proba(input_ids, attention_mask, token_type_ids)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

    return np.array(all_probs), np.array(all_labels)


# ============================================================
# TRAIN
# ============================================================

def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    patience: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train: {len(df_train):,} cặp | Val: {len(df_val):,} cặp")

    tokenizer = load_tokenizer(pretrained_name)

    train_dataset = PairwiseDataset(df_train, tokenizer, max_length)
    val_dataset = PairwiseDataset(df_val, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = CrossEncoderModel(pretrained_name=pretrained_name).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(output_dir, "tang4_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Lưu checkpoint tốt nhất: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> Early stopping (không cải thiện sau {patience} epoch)")
                break

    model.load_state_dict(torch.load(os.path.join(output_dir, "tang4_best.pt")))
    val_probs, val_labels = evaluate(model, val_loader, device)
    val_preds = (val_probs >= 0.5).astype(int)

    print(f"\n=== Báo cáo chi tiết (checkpoint tốt nhất, val_F1={best_val_f1:.4f}) ===")
    print(classification_report(val_labels, val_preds, target_names=["Không đạo văn", "Đạo văn"]))

    tokenizer.save_pretrained(output_dir)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", default="checkpoints/tang4")
    parser.add_argument("--pretrained-name", default=DEFAULT_PRETRAINED_NAME)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    train(
        train_csv=args.train, val_csv=args.val, output_dir=args.output_dir,
        pretrained_name=args.pretrained_name, max_length=args.max_length,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        num_epochs=args.num_epochs, patience=args.patience,
    )
