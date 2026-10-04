"""
Tầng 3 — Định nghĩa kiến trúc model (Bi-Encoder dựa trên Sentence-BERT)
==========================================================================

Khác với Tầng 0 (tự viết kiến trúc bằng torch.nn.Module từ đầu vì cần
custom classifier head), Tầng 3 tận dụng trực tiếp class
SentenceTransformer có sẵn trong thư viện sentence-transformers — vì
đây chính xác là kiến trúc Bi-Encoder cần dùng (có sẵn pooling layer,
sẵn sàng fine-tune bằng Triplet Loss), không cần tự viết lại.

File này là NƠI DUY NHẤT định nghĩa cấu hình kiến trúc (tên base
model...) — train.py và inference.py đều import từ đây, tránh lặp
code cấu hình ở nhiều nơi khác nhau.

GHI CHÚ VỀ KIẾN TRÚC "1 ENCODER DÙNG CHUNG" (đã thống nhất):
    Bản chất "dùng chung trọng số cho cả 3 nhánh Anchor/Positive/
    Negative" đến từ việc TripletLoss của sentence-transformers gọi
    CÙNG MỘT instance SentenceTransformer 3 lần (1 lần/câu) trong mỗi
    bước huấn luyện — không phải từ cấu hình đặc biệt nào ở file này.
    Đây là điểm khác biệt so với TriPlaNet (2025), vốn dùng 3 encoder
    riêng tham số (xem phần so sánh kiến trúc đã thống nhất trước đó).
"""

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DEFAULT_BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Model đa ngôn ngữ, 12 layer, hidden_size=384 — nhẹ, phù hợp fine-tune
# trên tài nguyên GPU hạn chế (Colab/Kaggle free tier). Có thể đổi
# sang bản lớn hơn ("paraphrase-multilingual-mpnet-base-v2") nếu có
# đủ tài nguyên và cần độ chính xác cao hơn.


def _check_library():
    if SentenceTransformer is None:
        raise ImportError(
            "Chưa cài thư viện sentence-transformers. Chạy: "
            "pip install sentence-transformers"
        )


def build_triplet_encoder(base_model: str = DEFAULT_BASE_MODEL) -> "SentenceTransformer":
    """
    Khởi tạo model Bi-Encoder TỪ BASE MODEL GỐC (chưa fine-tune) —
    dùng trong train.py để bắt đầu quá trình huấn luyện.
    """
    _check_library()
    print(f"Khởi tạo Bi-Encoder từ base model: {base_model}")
    return SentenceTransformer(base_model)


def load_finetuned_encoder(checkpoint_dir: str) -> "SentenceTransformer":
    """
    Load model Bi-Encoder ĐÃ FINE-TUNE từ checkpoint — dùng trong
    inference.py để suy luận, KHÔNG dùng để train tiếp.
    """
    _check_library()
    return SentenceTransformer(checkpoint_dir)
