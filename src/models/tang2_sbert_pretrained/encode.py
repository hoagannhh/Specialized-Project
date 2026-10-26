"""
Tầng 2 — So sánh ngữ nghĩa bằng encoder pretrained
=================================================

VAI TRÒ:
    Nhận cặp câu, tạo embedding độc lập cho từng câu và tính cosine
    similarity. Điểm này là đặc trưng ngữ nghĩa cho pipeline, không
    phải xác suất đạo văn. Luôn chuyển tiếp để chạy đủ các tầng sau.

KIẾN TRÚC — transformers + PyTorch:
    AutoTokenizer → AutoModel → Mean Pooling có attention mask
    → chuẩn hóa L2 → tích vô hướng của hai vector (cosine similarity).
    Pooling và chuẩn hóa được tự viết trong lớp BiEncoder; việc chia
    batch, chuyển thiết bị và tắt tính gradient nằm trong encode_texts.

PHÂN BIỆT VỚI TẦNG 3:
    Hai tầng dùng chung định nghĩa lớp BiEncoder, nhưng tạo instance
    riêng. Tầng 2 nạp trọng số pretrained gốc; tầng 3 nạp checkpoint
    đã fine-tune bằng Triplet Loss. Import lớp từ thư mục tầng 3
    không đồng nghĩa với nạp trọng số đã train của tầng 3.

    Tầng 2 không có bước train trong project. Backbone vẫn là model
    pretrained có sẵn, không phải Transformer được pretrain từ đầu.
    Tiền tố "sentence-transformers/" trong tên checkpoint là namespace
    trên Hugging Face; code không dùng thư viện sentence-transformers.

THAY ĐỔI SO VỚI LUỒNG CŨ:
    Không loại cặp khi cosine thấp, không kết luận sớm khi cosine cao.
    Các tham số ngưỡng cũ được giữ để tương thích lời gọi, không còn
    quyết định dừng. Kết quả decision luôn là "continue".
"""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
import torch
from transformers import AutoTokenizer
from src.models.runtime import EMBEDDING_MODEL, encode_texts, validate_pairs, batches
# Chỉ tái sử dụng định nghĩa kiến trúc PyTorch, không đọc checkpoint tầng 3.
from src.models.tang3_triplet_encoder.model import BiEncoder

# Cùng backbone khởi tạo với tầng 3 để so sánh trước/sau fine-tune.
DEFAULT_MODEL_NAME = EMBEDDING_MODEL
_cache = {}


def load_model(model_name=DEFAULT_MODEL_NAME, use_cache=True, device=None):
    """Nạp encoder pretrained và tokenizer; trả về tuple (model, tokenizer).

    Cache phân biệt tên model và thiết bị, tránh nạp lại ở mỗi cặp câu.
    eval() tắt dropout; encode_texts dùng inference_mode để không tạo
    đồ thị gradient khi suy luận.
    """
    device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    key = model_name, device
    if use_cache and key in _cache:
        return _cache[key]
    # BiEncoder gọi AutoModel.from_pretrained qua make_encoder trong runtime.py.
    result = BiEncoder(model_name).to(device).eval(), AutoTokenizer.from_pretrained(model_name)
    if use_cache:
        _cache.clear()
        _cache[key] = result
    return result


def _encode_batch(sentences, model, tokenizer, max_length=128, batch_size=32):
    """Mã hóa danh sách câu thành các vector đã chuẩn hóa L2, theo batch."""
    return encode_texts(sentences, model, tokenizer, max_length, batch_size)


def sbert_filter_batch(pairs, model=None, tokenizer=None, low_threshold=None,
                       high_threshold=None, batch_size=32, max_length=128):
    """Chấm điểm danh sách cặp (câu A, câu B), trả DataFrame kết quả.

    Chia batch để giới hạn bộ nhớ; max_length giới hạn số token mỗi câu.
    Danh sách rỗng trả bảng rỗng; câu rỗng hoặc cặp sai cấu trúc báo lỗi.
    low_threshold/high_threshold chỉ giữ để tương thích, không dùng lọc.
    """
    pairs = validate_pairs(pairs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not pairs:
        return pd.DataFrame(columns=['sentence_a', 'sentence_b', 'similarity', 'decision'])
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    scores = []
    for batch in batches(pairs, batch_size):
        a = encode_texts([p[0] for p in batch], model, tokenizer, max_length, batch_size)
        b = encode_texts([p[1] for p in batch], model, tokenizer, max_length, batch_size)
        # Vector đã chuẩn hóa L2 nên dot product chính là cosine similarity.
        # Tính từng cặp tương ứng, không tạo ma trận so sánh N × N.
        # Clamp chỉ xử lý sai số số học có thể vượt nhẹ ngoài [-1, 1].
        scores.extend((a * b).sum(dim=1).clamp(-1, 1).tolist())
    return pd.DataFrame({'sentence_a': [p[0] for p in pairs], 'sentence_b': [p[1] for p in pairs],
                         'similarity': scores, 'decision': 'continue'})


def sbert_filter(sentence_a, sentence_b, model=None, tokenizer=None,
                 low_threshold=None, high_threshold=None):
    """Chấm một cặp câu; trả cosine và yêu cầu tiếp tục các tầng sau.

    Tên hàm giữ từ phiên bản cũ, nhưng hiện không lọc hoặc kết luận nhãn.
    Tham số ngưỡng cũ không ảnh hưởng điểm hoặc decision.
    """
    row = sbert_filter_batch([(sentence_a, sentence_b)], model, tokenizer, batch_size=1).iloc[0]
    return {'similarity': float(row.similarity), 'decision': 'continue', 'score_type': 'cosine', 'reason': None}
