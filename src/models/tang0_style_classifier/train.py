"""
Tầng 0 — Script huấn luyện
============================

Áp dụng đúng các kỹ thuật đã thống nhất để xử lý mất cân bằng nhãn
(~82/18): class_weight trong loss, đánh giá bằng F1/PR-AUC (không
dùng Accuracy), tìm ngưỡng quyết định tối ưu trên tập validation.

Cách chạy:
    python src/models/tang0_style_classifier/train.py \\
        --train data/splits/tang0_train.csv \\
        --val data/splits/tang0_val.csv \\
        --output-dir checkpoints/tang0
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
    from transformers import AutoTokenizer
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report
except ImportError as e:
    raise ImportError(
        f"Thiếu thư viện cần thiết ({e}). Chạy: "
        f"pip install torch transformers scikit-learn"
    )

from src.models.tang0_style_classifier.model import SingleSentenceClassifier, DEFAULT_PRETRAINED_NAME


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


def evaluate(model, dataloader, device, check_time=None, loss_fn=None):
    """
    Đánh giá trên tập val/test, trả về xác suất dự đoán + nhãn thật
    (để sau đó tính F1, PR-AUC, hoặc tìm ngưỡng tối ưu bên ngoài).
    """
    model.eval()
    all_probs, all_labels = [], []
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            if check_time is not None:
                check_time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            if loss_fn is not None:
                losses.append(loss_fn(logits, labels.to(device)).item())
            all_probs.extend(probs[:, 1].cpu().numpy())  # xác suất lớp 1 (AI-touched)
            all_labels.extend(labels.numpy())

    result = (np.array(all_probs), np.array(all_labels))
    return (*result, float(np.mean(losses))) if loss_fn is not None else result


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Tìm ngưỡng quyết định cho F1 tốt nhất TRÊN TẬP VALIDATION.
    Ngưỡng này sau đó được áp dụng CỐ ĐỊNH khi test/suy luận thật —
    KHÔNG được tìm lại ngưỡng trên tập test (tránh rò rỉ thông tin).
    """
    return select_threshold(probs, labels)


@pause_cleanly
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
    seed: int = 42,
    resume_from: str = None,
    save_every_steps: int = 500,
    save_every_minutes: float = 10,
    max_runtime_minutes: float = None,
):
    if min(batch_size, max_length, num_epochs, patience) < 1:
        raise ValueError('batch_size, max_length, num_epochs and patience must be positive')
    run = TrainingRun(output_dir, 0, train_csv, val_csv,
                      dict(pretrained_name=str(pretrained_name), max_length=max_length, batch_size=batch_size,
                  learning_rate=learning_rate, patience=patience, seed=seed),
                      resume_from, save_every_steps, save_every_minutes, max_runtime_minutes)
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    if df_train.empty or df_val.empty:
        raise ValueError('Training and validation datasets must not be empty')
    print(f"Train: {len(df_train):,} dòng | Val: {len(df_val):,} dòng")

    for name, frame in [('train', df_train), ('val', df_val)]:
        if set(frame['label'].unique()) != {0, 1}:
            raise ValueError(f'{name} requires both binary labels 0 and 1')
    tokenizer = AutoTokenizer.from_pretrained(run.tokenizer_source)

    train_dataset = SingleSentenceDataset(df_train, tokenizer, max_length)
    val_dataset = SingleSentenceDataset(df_val, tokenizer, max_length)

    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                            generator=torch.Generator().manual_seed(seed))

    model = SingleSentenceClassifier(pretrained_name=pretrained_name, encoder_config=run.encoder_config).to(device)

    class_weights = compute_class_weights(df_train, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    run.attach(model, tokenizer, optimizer)
    best_val_f1 = run.state['best_metric']
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run.after_step(batch_index, loss.item())

        # Lưu trước validation: bị ngắt ở đây chỉ cần chạy lại validation.
        run.save()
        avg_train_loss = run.state['total_loss'] / run.state['next_batch']

        val_probs, val_labels, val_loss = evaluate(model, val_loader, device, check_time=run.check_time, loss_fn=loss_fn)
        val_threshold, _ = find_best_threshold(val_probs, val_labels)
        val_preds = (val_probs >= val_threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        precision, recall, _ = precision_recall_curve(val_labels, val_probs)
        pr_auc = auc(recall, precision)

        print(f"\n[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_F1(threshold={val_threshold:.4f})={val_f1:.4f} val_PR-AUC={pr_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(output_dir, "tang0_best.pt")
            save_bundle(model, tokenizer, output_dir, 0, pretrained_name, max_length, threshold=val_threshold, seed=seed, epoch=epoch, val_f1=float(val_f1))
            print(f"  -> Lưu checkpoint tốt nhất: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
        stopped = epochs_without_improvement >= patience
        run.finish_epoch(best_val_f1, epochs_without_improvement, stopped or epoch == num_epochs,
                         metrics=dict(train_loss=avg_train_loss, val_loss=val_loss,
                                      val_f1=float(val_f1), threshold=float(val_threshold), val_pr_auc=float(pr_auc)))
        if stopped:
            print(f"  -> Early stopping (không cải thiện sau {patience} epoch)")
            break

    # Load lại checkpoint tốt nhất, tìm ngưỡng tối ưu trên val set
    best_meta = restore_best(model, output_dir, 0)
    val_probs, val_labels = evaluate(model, val_loader, device, check_time=lambda: run.check_time(save=False))
    best_threshold, best_f1 = find_best_threshold(val_probs, val_labels)

    print(f"\n=== Ngưỡng quyết định tối ưu (tìm trên val set): {best_threshold:.2f} "
          f"(F1={best_f1:.4f}) ===")
    print("\nBáo cáo chi tiết với ngưỡng tối ưu:")
    val_preds_best = (val_probs >= best_threshold).astype(int)
    print(classification_report(val_labels, val_preds_best, target_names=["Người viết", "AI-touched"]))

    save_bundle(model, tokenizer, output_dir, 0, pretrained_name, max_length, threshold=best_threshold, seed=seed, **best_meta["training"])

    # Compatibility export; canonical threshold is stored inside the checkpoint.
    with open(os.path.join(output_dir, "threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    return model, best_threshold


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    parser.add_argument("--seed", type=int, default=42)
    add_checkpoint_arguments(parser)
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
        patience=args.patience, seed=args.seed,
        resume_from=args.resume_from, save_every_steps=args.save_every_steps,
        save_every_minutes=args.save_every_minutes, max_runtime_minutes=args.max_runtime_minutes,
    )
