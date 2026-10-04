"""
cascade_pipeline.py
---------------------
Ghép toàn bộ 5 tầng của CaTE-X thành 1 pipeline suy luận hoàn chỉnh,
CHẠY TUẦN TỰ với early-exit — đúng theo sơ đồ đã thống nhất:

    Input (A, B)
        -> Tầng 1 (TF-IDF)         [có thể dừng sớm]
        -> Tầng 2 (SBERT pretrained) [có thể dừng sớm]
        -> Tầng 3 (SBERT fine-tuned) [có thể dừng sớm nếu "confident"]
        -> Tầng 4 (Cross-Encoder)   [luôn chạy nếu đến đây]
        -> Tầng 5 (LLM-as-Judge)    [luôn chạy nếu đến đây, tùy chọn]
        -> Output

LƯU Ý QUAN TRỌNG (đã thống nhất nhiều lần): mỗi tầng nhận TRỰC TIẾP
văn bản thô (A, B), KHÔNG nhận output đã biến đổi của tầng trước —
các tầng chỉ phụ thuộc nhau về THỨ TỰ THỰC THI (control flow), không
phụ thuộc nhau về DỮ LIỆU (data flow).
"""

import importlib
import sys
import time
from pathlib import Path

# ============================================================
# LOAD MODULE THEO TỪNG THƯ MỤC TẦNG (xử lý đúng import nội bộ)
# ============================================================
# LƯU Ý QUAN TRỌNG: nhiều tầng có file TRÙNG TÊN (Tầng 3 và Tầng 4
# đều có "model.py" + "inference.py"; "inference.py" của mỗi tầng
# còn có dòng "from model import ..." — tức import NỘI BỘ dựa vào
# TÊN, không phải đường dẫn). Vì vậy:
#
#   - KHÔNG thể dùng sys.path.insert() thường (các thư mục tầng sẽ
#     đụng độ tên "model"/"inference" lẫn nhau)
#   - KHÔNG thể dùng importlib.util.spec_from_file_location() đơn
#     thuần (đã thử — "from model import ..." bên trong inference.py
#     sẽ báo lỗi "No module named 'model'", vì cách load này không
#     thêm thư mục chứa file vào sys.path)
#
# Giải pháp đã kiểm chứng: với MỖI tầng, tạm thời thêm ĐÚNG thư mục
# tầng đó vào đầu sys.path, xóa cache cũ (nếu có) của các tên
# "model"/"inference" trong sys.modules, import bình thường (lúc này
# "from model import ..." bên trong sẽ tìm đúng model.py CỦA TẦNG
# ĐÓ), rồi gỡ thư mục khỏi sys.path ngay sau đó. Module đã import
# được LƯU LẠI dưới biến riêng (ví dụ _tang3_inference), nên dù cache
# sys.modules bị dọn cho tầng sau, tầng này vẫn dùng được bình thường
# (không phụ thuộc cache sau khi đã lấy được tham chiếu module).

_SRC_DIR = Path(__file__).resolve().parents[1]


def _load_tier_module(tier_dir: Path, module_filename: str = "inference"):
    tier_dir_str = str(tier_dir)
    sys.path.insert(0, tier_dir_str)
    try:
        for name in ["model", module_filename]:
            sys.modules.pop(name, None)
        module = importlib.import_module(module_filename)
    finally:
        sys.path.remove(tier_dir_str)
    return module


_tang1_filter = _load_tier_module(_SRC_DIR / "models" / "tang1_tfidf", "filter")
_tang2_encode = _load_tier_module(_SRC_DIR / "models" / "tang2_sbert_pretrained", "encode")
_tang3_inference = _load_tier_module(_SRC_DIR / "models" / "tang3_triplet_encoder", "inference")
_tang4_inference = _load_tier_module(_SRC_DIR / "models" / "tang4_cross_encoder", "inference")
_tang5_call_api = _load_tier_module(_SRC_DIR / "models" / "tang5_llm_judge", "call_api")


class CaTEXPipeline:
    """
    Pipeline suy luận CaTE-X đầy đủ. Khởi tạo 1 LẦN (sẽ load/cache
    tất cả model cần thiết), sau đó gọi predict() nhiều lần cho
    từng cặp câu — tận dụng cache, không load lại model mỗi lần.
    """

    def __init__(
        self,
        tang1_vectorizer_path: str = "data/models/tang1/tfidf_vectorizer.pkl",
        tang2_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        tang3_checkpoint_dir: str = "checkpoints/tang3",
        tang4_checkpoint_dir: str = "checkpoints/tang4",
        enable_tang5: bool = False,
        tang5_provider: str = "openai",
        tang5_api_key: str = None,
        verbose: bool = True,
    ):
        self.tang1_vectorizer_path = tang1_vectorizer_path
        self.tang2_model_name = tang2_model_name
        self.tang3_checkpoint_dir = tang3_checkpoint_dir
        self.tang4_checkpoint_dir = tang4_checkpoint_dir
        self.enable_tang5 = enable_tang5
        self.tang5_provider = tang5_provider
        self.tang5_api_key = tang5_api_key
        self.verbose = verbose

        self._log("Khởi tạo pipeline — các model sẽ được load khi cần (lazy loading)")

    def _log(self, message: str):
        if self.verbose:
            print(f"[CaTE-X] {message}")

    def predict(self, sentence_a: str, sentence_b: str) -> dict:
        """
        Chạy pipeline đầy đủ cho 1 cặp câu, trả về kết quả cuối cùng
        kèm "trace" (lịch sử các tầng đã chạy qua) để dễ debug/giải
        thích trong demo.
        """
        trace = []
        start_time = time.time()

        # ---------------- TẦNG 1 ----------------
        t1_start = time.time()
        try:
            vectorizer = _tang1_filter.load_vectorizer(self.tang1_vectorizer_path)
            t1_result = _tang1_filter.tfidf_filter(sentence_a, sentence_b, vectorizer=vectorizer)
        except FileNotFoundError:
            self._log("CẢNH BÁO: chưa có vectorizer Tầng 1, bỏ qua Tầng 1 (coi như continue)")
            t1_result = {"similarity": None, "decision": "continue", "reason": "tang1_not_ready"}
        trace.append({"tang": 1, "result": t1_result, "time_ms": (time.time() - t1_start) * 1000})
        self._log(f"Tầng 1: similarity={t1_result.get('similarity')}, decision={t1_result['decision']}")

        if t1_result["decision"] == "stop":
            return self._finalize(trace, label="không đạo văn", probability=t1_result.get("similarity") or 0.0,
                                   stopped_at="Tầng 1", start_time=start_time)

        # ---------------- TẦNG 2 ----------------
        t2_start = time.time()
        t2_result = _tang2_encode.sbert_filter(sentence_a, sentence_b)
        trace.append({"tang": 2, "result": t2_result, "time_ms": (time.time() - t2_start) * 1000})
        self._log(f"Tầng 2: similarity={t2_result['similarity']:.3f}, decision={t2_result['decision']}")

        if t2_result["decision"] == "stop":
            return self._finalize(trace, label="không đạo văn", probability=t2_result["similarity"],
                                   stopped_at="Tầng 2", start_time=start_time)
        if t2_result["decision"] == "early_positive":
            return self._finalize(trace, label="đạo văn (AI-paraphrase)", probability=t2_result["similarity"],
                                   stopped_at="Tầng 2 (kết luận sớm)", start_time=start_time)

        # ---------------- TẦNG 3 ----------------
        t3_start = time.time()
        t3_result = _tang3_inference.tang3_predict(
            sentence_a, sentence_b, checkpoint_dir=self.tang3_checkpoint_dir
        )
        trace.append({"tang": 3, "result": t3_result, "time_ms": (time.time() - t3_start) * 1000})
        self._log(f"Tầng 3: similarity={t3_result['similarity']:.3f}, decision={t3_result['decision']}")

        if not t3_result["needs_tang4"]:
            label = "đạo văn (AI-paraphrase)" if t3_result["similarity"] >= 0.5 else "không đạo văn"
            return self._finalize(trace, label=label, probability=t3_result["similarity"],
                                   stopped_at="Tầng 3 (confident)", start_time=start_time)

        # ---------------- TẦNG 4 ----------------
        t4_start = time.time()
        t4_result = _tang4_inference.tang4_predict(
            sentence_a, sentence_b, checkpoint_dir=self.tang4_checkpoint_dir
        )
        trace.append({"tang": 4, "result": t4_result, "time_ms": (time.time() - t4_start) * 1000})
        self._log(f"Tầng 4: probability={t4_result['probability']:.3f}, label={t4_result['label']}")

        final_label = t4_result["label"]
        final_probability = t4_result["probability"]

        # ---------------- TẦNG 5 (tùy chọn) ----------------
        explanation = None
        if self.enable_tang5:
            t5_start = time.time()
            try:
                t5_result = _tang5_call_api.tang5_judge(
                    sentence_a, sentence_b,
                    model_label=final_label,
                    model_probability=final_probability,
                    provider=self.tang5_provider,
                    api_key=self.tang5_api_key,
                )
                trace.append({"tang": 5, "result": t5_result, "time_ms": (time.time() - t5_start) * 1000})
                explanation = t5_result.get("explanation")
                self._log(f"Tầng 5: explanation={explanation}")
            except Exception as e:
                self._log(f"CẢNH BÁO: Tầng 5 lỗi ({e}), bỏ qua, dùng kết quả Tầng 4")

        return self._finalize(
            trace, label=final_label, probability=final_probability,
            stopped_at="Tầng 4" + (" + Tầng 5" if explanation else ""),
            start_time=start_time, explanation=explanation,
        )

    def _finalize(self, trace, label, probability, stopped_at, start_time, explanation=None):
        total_time_ms = (time.time() - start_time) * 1000
        return {
            "label": label,
            "probability": probability,
            "stopped_at": stopped_at,
            "explanation": explanation,
            "total_time_ms": total_time_ms,
            "trace": trace,
        }


if __name__ == "__main__":
    pipeline = CaTEXPipeline(enable_tang5=False)

    result = pipeline.predict(
        "The economy grew rapidly last year.",
        "Last year, the economy expanded at a fast pace.",
    )

    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(f"Nhãn       : {result['label']}")
    print(f"Xác suất   : {result['probability']}")
    print(f"Dừng ở     : {result['stopped_at']}")
    print(f"Thời gian  : {result['total_time_ms']:.1f} ms")
