"""
retrieval/extract_page_text.py
---------------------------------
Trích xuất nội dung văn bản chính từ 1 trang web (loại bỏ HTML tag,
menu, quảng cáo, script...), dùng cho các URL trả về từ web_search.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import requests
from bs4 import BeautifulSoup

from config import WEB_SEARCH_TIMEOUT_SECONDS
from logger import get_logger

logger = get_logger(__name__)

# Các thẻ HTML chắc chắn KHÔNG chứa nội dung chính (menu, script, quảng cáo...)
TAGS_TO_REMOVE = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]

MIN_PARAGRAPH_LENGTH = 40  # Bỏ qua đoạn quá ngắn (thường là menu/caption, không phải nội dung thật)

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    # Một số trang web chặn request không có User-Agent giống trình duyệt thật
}


def fetch_and_extract_text(url: str, timeout: int = WEB_SEARCH_TIMEOUT_SECONDS) -> dict:
    """
    Tải 1 trang web và trích xuất nội dung văn bản chính.

    Returns
    -------
    dict: {"url": ..., "text": ..., "success": True/False, "error": None/str}

    QUAN TRỌNG: luôn trả về dict (KHÔNG raise exception) để
    single_file_pipeline.py có thể bỏ qua trang lỗi và thử trang
    tiếp theo trong danh sách kết quả tìm kiếm, thay vì dừng cả
    chương trình.
    """
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Không tải được trang {url}: {e}")
        return {"url": url, "text": "", "success": False, "error": str(e)}

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        logger.warning(f"Trang {url} không phải HTML (Content-Type: {content_type}), bỏ qua.")
        return {"url": url, "text": "", "success": False, "error": "not_html"}

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        logger.warning(f"Lỗi khi parse HTML của {url}: {e}")
        return {"url": url, "text": "", "success": False, "error": str(e)}

    for tag in soup(TAGS_TO_REMOVE):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text_parts = [
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if len(p.get_text(strip=True)) >= MIN_PARAGRAPH_LENGTH
    ]

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        logger.warning(f"Không trích xuất được nội dung có nghĩa từ {url}")
        return {"url": url, "text": "", "success": False, "error": "no_content_extracted"}

    logger.info(f"Trích xuất thành công {len(full_text):,} ký tự từ {url}")
    return {"url": url, "text": full_text, "success": True, "error": None}


def fetch_and_extract_batch(urls: list, timeout: int = WEB_SEARCH_TIMEOUT_SECONDS) -> list:
    """Xử lý nhiều URL liên tiếp, bỏ qua URL lỗi, không dừng cả batch."""
    return [fetch_and_extract_text(url, timeout=timeout) for url in urls]


if __name__ == "__main__":
    result = fetch_and_extract_text("https://github.com/about")
    print(f"Success: {result['success']}")
    print(f"Text preview ({len(result['text'])} ký tự):")
    print(result["text"][:400])

    print("\n--- Test với URL lỗi (không tồn tại) ---")
    bad_result = fetch_and_extract_text("https://github.com/this-page-does-not-exist-xyz123")
    print(f"Success: {bad_result['success']}, Error: {bad_result['error']}")
