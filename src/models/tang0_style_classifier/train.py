"""
Tầng 0 — Script huấn luyện
============================

Áp dụng đúng các kỹ thuật đã thống nhất để xử lý mất cân bằng nhãn
(~82/18): class_weight trong loss, đánh giá bằng F1/PR-AUC (không
dùng Accuracy), tìm ngưỡng quyết định tối ưu trên tập validation.

Cách chạy:
    python src/models/tang0_style_classifier/train.py \\
        --train data/splits/merged_tang0_train.csv \\
        --val data/splits/merged_tang0_val.csv \\
        --output-dir checkpoints/tang0
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report
except ImportError as e:
    raise ImportError(
        f"Thiếu thư viện cần thiết ({e}). Chạy: "
        f"pip install torch transformers scikit-learn"
    )

from model import SingleSentenceClassifier, DEFAULT_PRETRAINED_NAME


# ============================================================
# DATASET
# ============================================================

class SingleSentenceDataset(Dataset):
    """
    Dataset cho bài toán đơn-câu. Mỗi mẫu trả về (input_ids,
    attention_mask, label) đã token hóa sẵn.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.sentences = df["sentence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# TRAIN / EVAL LOOP
# ============================================================

def compute_class_weights(df_train: pd.DataFrame, device) -> "torch.Tensor":
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=df_train["label"].values,
    )
    print(f"Class weights tính được: label0={weights[0]:.3f}, label1={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(model, dataloader, device):
    """
    Đánh giá trên tập val/test, trả về xác suất dự đoán + nhãn thật
    (để sau đó tính F1, PR-AUC, hoặc tìm ngưỡng tối ưu bên ngoài).
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            probs = model.predict_proba(input_ids, attention_mask)
            all_probs.extend(probs[:, 1].cpu().numpy())  # xác suất lớp 1 (AI-touched)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Tìm ngưỡng quyết định cho F1 tốt nhất TRÊN TẬP VALIDATION.
    Ngưỡng này sau đó được áp dụng CỐ ĐỊNH khi test/suy luận thật —
    KHÔNG được tìm lại ngưỡng trên tập test (tránh rò rỉ thông tin).
    """
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold, best_f1


def train(
    train_csv: str,
    val_csv: str,
    output_dir: str,
    pretrained_name: str = DEFAULT_PRETRAINED_NAME,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 5,
    patience: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train: {len(df_train):,} dòng | Val: {len(df_val):,} dòng")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    train_dataset = SingleSentenceDataset(df_train, tokenizer, max_length)
    val_dataset = SingleSentenceDataset(df_val, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = SingleSentenceClassifier(pretrained_name=pretrained_name).to(device)

    class_weights = compute_class_weights(df_train, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
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

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_probs, val_labels = evaluate(model, val_loader, device)
        val_preds_05 = (val_probs >= 0.5).astype(int)
        val_f1_at_05 = f1_score(val_labels, val_preds_05, zero_division=0)

        precision, recall, _ = precision_recall_curve(val_labels, val_probs)
        pr_auc = auc(recall, precision)

        print(f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_F1(ngưỡng 0.5)={val_f1_at_05:.4f} val_PR-AUC={pr_auc:.4f}")

        if val_f1_at_05 > best_val_f1:
            best_val_f1 = val_f1_at_05
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(output_dir, "tang0_best.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Lưu checkpoint tốt nhất: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> Early stopping (không cải thiện sau {patience} epoch)")
                break

    # Load lại checkpoint tốt nhất, tìm ngưỡng tối ưu trên val set
    model.load_state_dict(torch.load(os.path.join(output_dir, "tang0_best.pt")))
    val_probs, val_labels = evaluate(model, val_loader, device)
    best_threshold, best_f1 = find_best_threshold(val_probs, val_labels)

    print(f"\n=== Ngưỡng quyết định tối ưu (tìm trên val set): {best_threshold:.2f} "
          f"(F1={best_f1:.4f}) ===")
    print("\nBáo cáo chi tiết với ngưỡng tối ưu:")
    val_preds_best = (val_probs >= best_threshold).astype(int)
    print(classification_report(val_labels, val_preds_best, target_names=["Người viết", "AI-touched"]))

    # Lưu lại ngưỡng để dùng ở inference.py
    with open(os.path.join(output_dir, "threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    return model, best_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", default="checkpoints/tang0")
    parser.add_argument("--pretrained-name", default=DEFAULT_PRETRAINED_NAME)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    train(
        train_csv=args.train,
        val_csv=args.val,
        output_dir=args.output_dir,
        pretrained_name=args.pretrained_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
    )
