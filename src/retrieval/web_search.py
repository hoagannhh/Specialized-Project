"""
retrieval/web_search.py
-------------------------
Gọi Google Custom Search API để tìm nguồn nghi ngờ cho 1 câu đã
được Bước 0 đánh dấu "đáng nghi" (xem sơ đồ hệ thống mở rộng
input-1-file: Bước 0 -> Web Search -> Tầng 1-5).
"""

import sys
from pathlib import Path

# utils/ có tên file DUY NHẤT trong toàn project (config.py, logger.py
# không trùng tên với bất kỳ module nào khác) -> an toàn khi thêm vào
# sys.path trực tiếp, không cần kỹ thuật importlib phức tạp như khi
# xử lý các tầng có file trùng tên (model.py, inference.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import requests

from config import (
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_ENGINE_ID,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_DAILY_QUOTA_WARNING,
    WEB_SEARCH_TIMEOUT_SECONDS,
)
from logger import get_logger

logger = get_logger(__name__)

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

_call_count = 0  # đếm số lần gọi trong phiên chạy hiện tại, để cảnh báo gần hạn mức


def search_web(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list:
    """
    Tìm kiếm trên web bằng Google Custom Search API.

    Returns
    -------
    list các dict: [{"title": ..., "url": ..., "snippet": ...}, ...]

    QUAN TRỌNG: hàm này trả về list RỖNG (KHÔNG raise exception) nếu
    có lỗi — để single_file_pipeline.py có thể tiếp tục xử lý câu
    tiếp theo trong file, thay vì dừng toàn bộ chương trình chỉ vì
    1 lần gọi API thất bại (mạng chập chờn, 1 câu query lỗi...).
    """
    global _call_count

    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        logger.error("Thiếu GOOGLE_SEARCH_API_KEY hoặc GOOGLE_SEARCH_ENGINE_ID, bỏ qua tìm kiếm.")
        return []

    if not query or not query.strip():
        return []

    _call_count += 1
    if _call_count >= WEB_SEARCH_DAILY_QUOTA_WARNING:
        logger.warning(
            f"Đã gọi {_call_count} lần trong phiên này — gần chạm hạn mức "
            f"miễn phí (100 lượt/ngày). Cân nhắc dừng lại hoặc nâng cấp gói trả phí."
        )

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": min(max_results, 10),  # Google giới hạn tối đa 10 kết quả/lần gọi
    }

    try:
        response = requests.get(GOOGLE_SEARCH_ENDPOINT, params=params, timeout=WEB_SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi gọi Google Search API: {e}")
        return []

    try:
        data = response.json()
    except ValueError:
        logger.error("Phản hồi từ Google Search API không phải JSON hợp lệ.")
        return []

    if "error" in data:
        logger.error(f"Google Search API trả về lỗi: {data['error'].get('message', data['error'])}")
        return []

    items = data.get("items", [])
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in items
    ]

    logger.info(f"Tìm thấy {len(results)} kết quả cho query: '{query[:60]}...'")
    return results


if __name__ == "__main__":
    results = search_web("artificial intelligence changing education")
    for r in results:
        print(r)
