"""
utils/module_loader.py
-------------------------
Nạp module của từng tầng theo ĐÚNG thư mục của tầng đó.

Vì sao cần: nhiều tầng có file TRÙNG TÊN (Tầng 0, 3, 4 đều có
"model.py" + "inference.py"; "inference.py" lại có dòng
"from model import ..."). Import thường qua sys.path sẽ bị đụng tên
(đã kiểm chứng bằng thực nghiệm khi viết cascade_pipeline.py).

Cách làm: với MỖI lần nạp, tạm thời thêm ĐÚNG thư mục tầng vào đầu
sys.path, xóa cache sys.modules của các tên "model"/<module> (nếu có
từ tầng trước), import, rồi gỡ thư mục khỏi sys.path. Module đã nạp
được giữ bằng tham chiếu của người gọi nên vẫn dùng được bình thường
dù cache bị dọn cho tầng sau.

GHI CHÚ: cascade_pipeline.py và single_file_pipeline.py hiện có bản
sao riêng của hàm này (đã chạy ổn). Có thể chuyển sang dùng file này
để bớt trùng lặp, nhưng không bắt buộc.
"""

import importlib
import sys
from pathlib import Path


def load_tier_module(tier_dir, module_filename: str = "inference"):
    """
    Parameters
    ----------
    tier_dir : thư mục chứa module (ví dụ src/models/tang3_triplet_encoder)
    module_filename : tên file KHÔNG có đuôi .py (ví dụ "inference")
    """
    tier_dir = Path(tier_dir)
    if not tier_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục tầng: {tier_dir}")

    tier_dir_str = str(tier_dir)
    sys.path.insert(0, tier_dir_str)
    try:
        for name in ["model", module_filename]:
            sys.modules.pop(name, None)
        module = importlib.import_module(module_filename)
    finally:
        sys.path.remove(tier_dir_str)
        # Dọn cache để lần nạp tầng kế tiếp không dính module của tầng này
        for name in ["model", module_filename]:
            sys.modules.pop(name, None)

    return module
