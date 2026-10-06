"""
utils/logger.py
-----------------
Thiết lập logging thống nhất cho toàn bộ project — dùng thay cho
print() rải rác, giúp kiểm soát mức độ chi tiết (DEBUG/INFO/WARNING/
ERROR) và dễ ghi log ra file khi chạy pipeline dài hơi (ví dụ xử lý
1 file có hàng trăm câu, cần lưu lại lịch sử để debug sau).
"""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """
    Tạo (hoặc lấy lại) logger theo tên module gọi hàm này.

    Cách dùng chuẩn trong mỗi file:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Bắt đầu xử lý...")
        logger.warning("Cảnh báo: ...")
        logger.error("Lỗi: ...")

    Parameters
    ----------
    log_file : nếu truyền vào, log sẽ GHI THÊM ra file này (ngoài in
               ra console) — hữu ích khi xử lý 1 file dài, muốn lưu
               lại lịch sử xử lý để debug sau.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger đã được cấu hình trước đó (ví dụ module được import
        # nhiều lần) -> không thêm handler trùng lặp, tránh log bị in lặp
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False  # tránh log bị in lặp lại qua root logger

    return logger
