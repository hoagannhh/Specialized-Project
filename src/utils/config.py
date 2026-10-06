"""
utils/config.py
-----------------
Cấu hình tập trung cho toàn bộ hệ thống: đường dẫn, ngưỡng quyết
định của từng tầng, API key (đọc từ biến môi trường / file .env).

TẤT CẢ module khác nên import config từ đây thay vì hardcode đường
dẫn hoặc đọc os.environ rải rác ở nhiều nơi — giúp dễ bảo trì, dễ
đổi cấu hình khi chuyển môi trường (local/Colab/server).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # Đọc file .env ở thư mục gốc project nếu có
except ImportError:
    pass  # Không bắt buộc phải có python-dotenv, chỉ là tiện lợi


# ============================================================
# ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]  # thư mục gốc project

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CROSS_EVAL_DIR = DATA_DIR / "cross_eval"
MODELS_DIR = DATA_DIR / "models"

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TANG0_CHECKPOINT_DIR = CHECKPOINTS_DIR / "tang0"
TANG3_CHECKPOINT_DIR = CHECKPOINTS_DIR / "tang3"
TANG4_CHECKPOINT_DIR = CHECKPOINTS_DIR / "tang4"

TANG1_VECTORIZER_PATH = MODELS_DIR / "tang1" / "tfidf_vectorizer.pkl"

EVALUATION_DIR = BASE_DIR / "evaluation"
EDA_RESULTS_DIR = EVALUATION_DIR / "eda_results"


# ============================================================
# NGƯỠNG QUYẾT ĐỊNH CỦA TỪNG TẦNG (đã thống nhất trong thiết kế)
# ============================================================

TANG1_THRESHOLD = 0.1          # Loại bỏ nếu similarity < ngưỡng này
TANG2_LOW_THRESHOLD = 0.2      # Dưới ngưỡng -> dừng
TANG2_HIGH_THRESHOLD = 0.85    # Trên ngưỡng -> kết luận sớm
TANG3_AMBIGUOUS_LOW = 0.4      # Vùng mập mờ: cần Tầng 4 tinh chỉnh
TANG3_AMBIGUOUS_HIGH = 0.85
TANG4_THRESHOLD = 0.5          # Ngưỡng phân loại cuối (đạo văn / không)

# Ngưỡng Bước 0 KHÔNG hardcode ở đây — được TÌM TỰ ĐỘNG lúc train
# (xem tang0/train.py) và lưu trong checkpoints/tang0/threshold.txt,
# vì ngưỡng tối ưu phụ thuộc vào tỷ lệ mất cân bằng dữ liệu thật.


# ============================================================
# API KEYS (đọc từ biến môi trường, xem .env.example)
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")


# ============================================================
# CẤU HÌNH TẦNG 5 (LLM-as-a-Judge)
# ============================================================

TANG5_PROVIDER = os.environ.get("TANG5_PROVIDER", "openai")  # "openai" hoặc "anthropic"
TANG5_ENABLED_BY_DEFAULT = False  # Tốn API call -> mặc định tắt, bật thủ công khi cần


# ============================================================
# CẤU HÌNH WEB SEARCH (hệ thống mở rộng input-1-file)
# ============================================================

WEB_SEARCH_MAX_RESULTS = 3           # Số trang web lấy về cho mỗi câu nghi ngờ
WEB_SEARCH_DAILY_QUOTA_WARNING = 90  # Cảnh báo khi gần chạm hạn mức free (100/ngày)
WEB_SEARCH_TIMEOUT_SECONDS = 10      # Timeout khi tải nội dung 1 trang web


def ensure_all_dirs():
    """Tạo sẵn toàn bộ thư mục cần thiết nếu chưa tồn tại."""
    for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CROSS_EVAL_DIR, MODELS_DIR,
              CHECKPOINTS_DIR, EVALUATION_DIR, EDA_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_required_keys(require_tang5: bool = False, require_web_search: bool = False):
    """
    Kiểm tra API key cần thiết đã được cấu hình chưa — gọi hàm này
    ở ĐẦU single_file_pipeline.py để báo lỗi SỚM và RÕ RÀNG, thay vì
    để pipeline chạy nửa chừng (có thể đã xử lý xong 50 câu) rồi mới
    crash vì thiếu key ở câu thứ 51.
    """
    missing = []

    if require_tang5:
        if TANG5_PROVIDER == "openai" and not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY (cần cho Tầng 5 với provider=openai)")
        elif TANG5_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY (cần cho Tầng 5 với provider=anthropic)")

    if require_web_search:
        if not GOOGLE_SEARCH_API_KEY:
            missing.append("GOOGLE_SEARCH_API_KEY (cần cho bước Web Search)")
        if not GOOGLE_SEARCH_ENGINE_ID:
            missing.append("GOOGLE_SEARCH_ENGINE_ID (cần cho bước Web Search)")

    if missing:
        raise EnvironmentError(
            "Thiếu cấu hình bắt buộc:\n  - " + "\n  - ".join(missing) +
            "\n\nHãy tạo file .env (sao chép từ .env.example) và điền đầy đủ."
        )
