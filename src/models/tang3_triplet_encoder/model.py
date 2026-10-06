"""
Tầng 3 — Kiến trúc Bi-Encoder (tự xây bằng PyTorch thuần)
=============================================================

KHÁC với phiên bản trước (dùng sentence-transformers.SentenceTransformer
có sẵn) — phiên bản này TỰ VIẾT kiến trúc từ đầu, giống phong cách
Tầng 0, để:
    1. Dễ giải thích/bảo vệ trước hội đồng (code nằm ngay trong đồ
       án, không "ẩn" trong thư viện ngoài)
    2. Dễ tùy biến (đổi pooling strategy, thêm layer...) nếu cần

Kiến trúc: 1 BERT encoder DÙNG CHUNG trọng số cho cả 3 nhánh
Anchor/Positive/Negative (gọi forward() 3 lần trên CÙNG 1 model) +
Mean Pooling (không chỉ lấy [CLS]) để tạo embedding câu.

TẠI SAO DÙNG MEAN POOLING THAY VÌ CHỈ LẤY [CLS] (khác Tầng 0 và
Tầng 4 — 2 tầng đó dùng [CLS] vì là bài toán PHÂN LOẠI, có lớp
classifier học cách "đọc" [CLS] theo mục đích phân loại cụ thể):
    Tầng 3 cần tạo ra embedding CÓ Ý NGHĨA HÌNH HỌC để so sánh
    similarity trực tiếp (không qua classifier nào thêm), nên cần
    pooling toàn bộ thông tin câu. Đây đúng là cách Sentence-BERT
    gốc (Reimers & Gurevych, 2019) đề xuất và đã được chứng minh
    hiệu quả hơn chỉ dùng [CLS] cho bài toán similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


DEFAULT_PRETRAINED_NAME = "bert-base-multilingual-cased"


class BiEncoder(nn.Module):
    """
    Model Tầng 3: nhận 1 câu (input_ids/attention_mask đã token hóa),
    trả về 1 vector embedding (đã chuẩn hóa L2) đại diện cho câu đó.

    QUAN TRỌNG: đây là encoder DUY NHẤT — khi train bằng Triplet
    Loss, cả 3 câu Anchor/Positive/Negative đều được đưa qua CÙNG
    instance model này (gọi forward() 3 lần), nên tự động "dùng
    chung trọng số" mà không cần thiết kế gì đặc biệt thêm.
    """

    def __init__(self, pretrained_name: str = DEFAULT_PRETRAINED_NAME):
        super().__init__()

        if AutoModel is None:
            raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")

        self.encoder = AutoModel.from_pretrained(pretrained_name)

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Lấy trung bình embedding của TẤT CẢ token thật (bỏ qua token
        padding), KHÔNG chỉ lấy [CLS] — đây là "Mean Pooling" theo
        đúng cách Sentence-BERT gốc thiết kế.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # tránh chia cho 0
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        embedding : torch.Tensor, shape (batch_size, hidden_size)
                    đã chuẩn hóa L2 (norm=1), để dot product giữa 2
                    embedding chính là cosine similarity.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        sentence_embedding = self._mean_pooling(token_embeddings, attention_mask)
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding


class TripletLoss(nn.Module):
    """
    Hàm mất mát Triplet Loss — đúng công thức đã trình bày trong
    Chương 3:
        L(A, P, N) = max(0, d(A,P) - d(A,N) + margin)

    Dùng cosine DISTANCE (= 1 - cosine similarity) làm d(.,.), vì
    embedding đã chuẩn hóa L2 ở BiEncoder phía trên.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Embedding đã chuẩn hóa L2 -> cosine similarity = dot product
        sim_ap = torch.sum(anchor_emb * positive_emb, dim=1)
        sim_an = torch.sum(anchor_emb * negative_emb, dim=1)

        dist_ap = 1 - sim_ap
        dist_an = 1 - sim_an

        losses = F.relu(dist_ap - dist_an + self.margin)

        return losses.mean()


def load_tokenizer(pretrained_name: str = DEFAULT_PRETRAINED_NAME):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Chưa cài thư viện transformers. Chạy: pip install transformers")
    return AutoTokenizer.from_pretrained(pretrained_name)
