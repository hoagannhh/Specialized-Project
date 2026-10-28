"""Vẽ lại lịch sử từ CSV đã lưu, không nạp model và không cần GPU."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(checkpoint_dir, smoothing_window=100):
    """Trả về figure; lưu PNG cạnh CSV để tải cùng checkpoint."""
    if smoothing_window < 1:
        raise ValueError('smoothing_window phải lớn hơn 0')
    root = Path(checkpoint_dir)
    steps_path, epochs_path = root / 'steps_history.csv', root / 'epochs_history.csv'
    steps = pd.read_csv(steps_path) if steps_path.exists() else pd.DataFrame()
    epochs = pd.read_csv(epochs_path) if epochs_path.exists() else pd.DataFrame()
    if steps.empty and epochs.empty:
        print('Chưa có lịch sử loss đã lưu. Chạy train hoặc chọn thư mục checkpoint mới.')
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if not steps.empty:
        axes[0].plot(steps.global_step, steps.train_loss, alpha=0.2, label='Batch loss gốc')
        smooth = steps.train_loss.rolling(smoothing_window, min_periods=1).mean()
        axes[0].plot(steps.global_step, smooth, label=f'Trung bình trượt {smoothing_window} bước')
        axes[0].legend()
    axes[0].set(title='Train loss theo bước', xlabel='Global step', ylabel='Loss')
    if not epochs.empty:
        axes[1].plot(epochs.epoch, epochs.train_loss, marker='o', label='Train')
        axes[1].plot(epochs.epoch, epochs.val_loss, marker='o', label='Validation')
        axes[1].legend()
        for column, label in [('val_f1', 'F1'), ('val_pr_auc', 'PR-AUC'),
                              ('val_accuracy', 'Triplet ranking accuracy')]:
            if column in epochs and epochs[column].notna().any():
                axes[2].plot(epochs.epoch, epochs[column], marker='o', label=label)
        axes[2].legend()
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, 'Chưa hoàn tất validation đầu tiên', ha='center', transform=ax.transAxes)
    axes[1].set(title='Loss theo epoch (trung bình batch)', xlabel='Epoch', ylabel='Loss')
    axes[2].set(title='Chỉ số validation', xlabel='Epoch', ylabel='Score', ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = root / 'training_curves.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    print('Đã lưu biểu đồ:', path)
    return fig
