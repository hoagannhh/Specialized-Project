"""
Tầng 4 — Kiến trúc Cross-Encoder (tự xây bằng PyTorch thuần)
================================================================

Kiến trúc: ghép 2 câu thành 1 chuỗi duy nhất theo định dạng
[CLS] câu_1 [SEP] câu_2 [SEP], đưa qua BERT encoder, lấy embedding
[CLS] (đại diện cho MỐI QUAN HỆ giữa 2 câu, vì self-attention đã
cho phép 2 câu "nhìn thấy nhau"), qua 1 lớp Linear duy nhất (1 đầu
ra) để phân loại nhị phân qua sigmoid.

Việc ghép 2 câu được transformers.AutoTokenizer tự xử lý khi gọi
tokenizer(sentence_1, sentence_2, ...) — tự thêm [CLS]/[SEP]/
token_type_ids đúng chuẩn BERT, không cần tự ghép tay.
"""

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


DEFAULT_PRETRAINED_NAME = "bert-base-multilingual-cased"


class CrossEncoderModel(nn.Module):
    """
    Model Tầng 4: nhận 1 CHUỖI ĐÃ GHÉP (input_ids/attention_mask/
    token_type_ids của cặp câu), trả về 1 logit (chưa qua sigmoid).

    num_labels=1 (khác Tầng 0 dùng num_classes=2) vì đây là phân
    loại nhị phân qua 1 giá trị sigmoid duy nhất — dùng với
    nn.BCEWithLogitsLoss (tự áp dụng sigmoid bên trong khi tính loss,
    ổn định số học hơn tự gọi sigmoid() rồi mới tính BCE thường).
    """

    def __init__(self, pretrained_name: str = DEFAULT_PRETRAINED_NAME, dropout: float = 0.1):
        super().__init__()

        if AutoModel is None:
            raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")

        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        logit : torch.Tensor, shape (batch_size,) — logit THÔ, dùng
                trực tiếp với BCEWithLogitsLoss lúc train, hoặc qua
                sigmoid() lúc suy luận để ra xác suất.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)

        logit = self.classifier(cls_embedding).squeeze(-1)

        return logit

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        """Trả về xác suất (đã qua sigmoid) — dùng lúc suy luận."""
        self.eval()
        logit = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logit)


def load_tokenizer(pretrained_name: str = DEFAULT_PRETRAINED_NAME):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")
    return AutoTokenizer.from_pretrained(pretrained_name)
