"""
pipeline/calibration.py
--------------------------
Platt Scaling — chuyển similarity score (Tầng 1, 2, 3) thành xác
suất THẬT SỰ có ý nghĩa thống kê, thay vì chỉ là độ đo khoảng cách
hình học (đã giải thích chi tiết lý do cần bước này khi thiết kế
output của hệ thống).

CẦN 1 BỘ CALIBRATOR RIÊNG CHO MỖI TẦNG (1, 2, 3) — vì "tính chất"
similarity của mỗi tầng khác nhau (TF-IDF khác SBERT pretrained khác
SBERT fine-tuned). Tầng 4 KHÔNG cần calibration vì đã xuất ra xác
suất chuẩn sẵn qua Binary Cross-Entropy + sigmoid.

Đây là thành phần TÙY CHỌN (nice-to-have) — hệ thống vẫn hoạt động
đúng mà không cần bước này, chỉ là "%" hiển thị ở Tầng 1-3 sẽ mang
ý nghĩa "similarity score" thay vì "xác suất đã hiệu chỉnh".
"""

import pickle
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
except ImportError as e:
    raise ImportError(f"Thiếu thư viện: {e}. Chạy: pip install scikit-learn")


def fit_calibrator(
    similarity_scores: np.ndarray,
    true_labels: np.ndarray,
    save_path: str = None,
) -> LogisticRegression:
    """
    Huấn luyện bộ hiệu chỉnh Platt Scaling: 1 Logistic Regression
    đơn giản, input là 1 similarity score, output là xác suất.

    Parameters
    ----------
    similarity_scores : mảng 1 chiều, similarity thô từ Tầng 1/2/3
    true_labels : mảng 1 chiều, nhãn thật (0/1) tương ứng
    """
    similarity_scores = np.asarray(similarity_scores).reshape(-1, 1)
    true_labels = np.asarray(true_labels)

    if len(np.unique(true_labels)) < 2:
        raise ValueError(
            "Dữ liệu chỉ có 1 lớp nhãn duy nhất, không thể fit calibrator. "
            "Cần cả nhãn 0 và 1 trong dữ liệu truyền vào."
        )

    calibrator = LogisticRegression()
    calibrator.fit(similarity_scores, true_labels)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(calibrator, f)
        print(f"Đã lưu calibrator vào: {save_path}")

    return calibrator


def load_calibrator(path: str) -> LogisticRegression:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy calibrator tại: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_calibration(similarity_score: float, calibrator: LogisticRegression) -> float:
    """
    Chuyển 1 similarity score thô thành xác suất đã hiệu chỉnh.
    """
    score_array = np.array([[similarity_score]])
    probability = calibrator.predict_proba(score_array)[0][1]
    return float(probability)


def evaluate_calibration(
    similarity_scores: np.ndarray,
    true_labels: np.ndarray,
    calibrator: LogisticRegression,
    n_bins: int = 10,
) -> dict:
    """
    Đánh giá chất lượng hiệu chỉnh bằng Brier Score và dữ liệu cho
    Reliability Diagram — dùng cho Chương 4 (phần đánh giá định lượng).

    Returns
    -------
    dict gồm: brier_score, fraction_of_positives, mean_predicted_value
    (2 mảng cuối dùng để vẽ Reliability Diagram)
    """
    similarity_scores = np.asarray(similarity_scores).reshape(-1, 1)
    true_labels = np.asarray(true_labels)

    calibrated_probs = calibrator.predict_proba(similarity_scores)[:, 1]

    brier = brier_score_loss(true_labels, calibrated_probs)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, calibrated_probs, n_bins=n_bins, strategy="uniform"
    )

    return {
        "brier_score": float(brier),
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
    }


if __name__ == "__main__":
    # Test nhanh với dữ liệu giả lập: similarity cao -> thường label=1,
    # similarity thấp -> thường label=0 (có nhiễu để giống thực tế)
    rng = np.random.RandomState(42)
    n = 500

    true_labels_demo = rng.randint(0, 2, size=n)
    # similarity tương quan với label nhưng có nhiễu
    similarity_demo = np.clip(
        true_labels_demo * 0.5 + rng.normal(0.3, 0.2, size=n), 0, 1
    )

    calibrator_demo = fit_calibrator(similarity_demo, true_labels_demo)

    print("Ví dụ áp dụng calibration:")
    for test_score in [0.1, 0.3, 0.5, 0.7, 0.9]:
        prob = apply_calibration(test_score, calibrator_demo)
        print(f"  similarity={test_score} -> xác suất đã hiệu chỉnh={prob:.3f}")

    eval_result = evaluate_calibration(similarity_demo, true_labels_demo, calibrator_demo)
    print(f"\nBrier score: {eval_result['brier_score']:.4f} (càng gần 0 càng tốt)")
