"""
Tầng 4 — Định nghĩa kiến trúc model (Cross-Encoder)
======================================================

Tận dụng trực tiếp class CrossEncoder có sẵn trong sentence-transformers
— đã tích hợp sẵn: ghép [CLS] câu_1 [SEP] câu_2 [SEP], BERT encoder,
classifier head, và BCEWithLogitsLoss khi num_labels=1. Không cần tự
viết nn.Module từ đầu như Tầng 0.

File này là NƠI DUY NHẤT định nghĩa cấu hình kiến trúc — train.py và
inference.py đều import từ đây.
"""

try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    CrossEncoder = None


DEFAULT_BASE_MODEL = "bert-base-multilingual-cased"
DEFAULT_MAX_LENGTH = 256


def _check_library():
    if CrossEncoder is None:
        raise ImportError(
            "Chưa cài thư viện sentence-transformers. Chạy: "
            "pip install sentence-transformers"
        )


def build_cross_encoder(
    base_model: str = DEFAULT_BASE_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> "CrossEncoder":
    """
    Khởi tạo Cross-Encoder TỪ BASE MODEL GỐC (chưa fine-tune) — dùng
    trong train.py.

    num_labels=1 -> CrossEncoder tự dùng BCEWithLogitsLoss (phân loại
    nhị phân qua 1 giá trị sigmoid duy nhất), đúng công thức đã thiết
    kế trong Chương 3 (Binary Cross-Entropy).
    """
    _check_library()
    print(f"Khởi tạo Cross-Encoder từ base model: {base_model}")
    return CrossEncoder(base_model, num_labels=1, max_length=max_length)


def load_finetuned_cross_encoder(checkpoint_dir: str) -> "CrossEncoder":
    """
    Load Cross-Encoder ĐÃ FINE-TUNE từ checkpoint — dùng trong
    inference.py, KHÔNG dùng để train tiếp.
    """
    _check_library()
    return CrossEncoder(checkpoint_dir)
