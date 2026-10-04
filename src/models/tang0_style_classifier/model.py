"""
Tầng 0 — Kiến trúc model (BERT/RoBERTa encoder + Linear classifier)
======================================================================

Kiến trúc: Encoder-only Transformer + 1 lớp tuyến tính gắn trên
embedding của token [CLS], phân loại nhị phân:
    label 0 = câu do người viết
    label 1 = câu đã qua AI paraphrase/sinh

Đây chính là kiến trúc Wahle et al. (2021) đã dùng, áp dụng cho bài
toán đơn-câu (không phải cặp câu).
"""

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


DEFAULT_PRETRAINED_NAME = "bert-base-multilingual-cased"
# Model đa ngôn ngữ (hỗ trợ tiếng Việt), phù hợp vì dataset có thể mở
# rộng đa ngôn ngữ sau này. Lựa chọn thay thế nhẹ/nhanh hơn (chỉ
# tiếng Anh): "bert-base-uncased" hoặc "distilbert-base-uncased"


class SingleSentenceClassifier(nn.Module):
    """
    Model Bước 0: nhận 1 câu (dạng input_ids/attention_mask đã token
    hóa sẵn), trả về logits cho 2 lớp (người viết / AI-touched).
    """

    def __init__(
        self,
        pretrained_name: str = DEFAULT_PRETRAINED_NAME,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if AutoModel is None:
            raise ImportError(
                "Chưa cài thư viện transformers. Chạy: pip install transformers"
            )

        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Trả về logits THÔ (chưa qua softmax) — dùng trực tiếp với
        nn.CrossEntropyLoss (hàm này tự áp dụng softmax bên trong).

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, num_classes)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Lấy embedding của token [CLS] — luôn là token đầu tiên (vị trí 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        return logits

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Trả về xác suất (đã qua softmax), dùng lúc suy luận (inference),
        KHÔNG dùng lúc train (train dùng forward() + CrossEntropyLoss).
        """
        self.eval()
        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        return probs
