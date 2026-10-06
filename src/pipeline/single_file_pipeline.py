"""
pipeline/single_file_pipeline.py
-----------------------------------
Hệ thống mở rộng: nhận 1 file văn bản duy nhất, tự động:
    1. Đọc file + tách câu
    2. Lọc sơ bộ bằng Bước 0 (chỉ câu "đáng nghi" mới đi tiếp)
    3. Tìm nguồn nghi ngờ qua Web Search
    4. Trích xuất nội dung trang web, tìm câu khớp nhất trong trang đó
    5. Đưa cặp (câu nghi ngờ, câu nguồn tìm được) qua cascade_pipeline
       (Tầng 1-5, KHÔNG đổi)
    6. Tổng hợp kết quả, xuất file HTML có highlight

Xem sơ đồ "Kiến trúc hệ thống mở rộng CaTE-X" đã thống nhất.

Cách chạy:
    python src/pipeline/single_file_pipeline.py \\
        --input path/to/document.docx \\
        --output result_highlighted.html
"""

import importlib
import re
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC_DIR / "utils"))

from config import ensure_all_dirs, check_required_keys, TANG0_CHECKPOINT_DIR, WEB_SEARCH_MAX_RESULTS
from logger import get_logger

logger = get_logger(__name__)


# ============================================================
# LOAD MODULE CẦN THIẾT (dùng lại kỹ thuật từ cascade_pipeline.py để
# xử lý đúng các file trùng tên giữa các tầng — xem giải thích chi
# tiết trong cascade_pipeline.py)
# ============================================================

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


def _load_modules():
    """
    Load trễ (lazy) — chỉ gọi khi thực sự cần khởi tạo pipeline, để
    các hàm thuần túy (read_file, split_sentences...) trong file này
    vẫn import/test được ngay cả khi CHƯA cài torch/sentence-transformers.
    """
    tang0_inference = _load_tier_module(_SRC_DIR / "models" / "tang0_style_classifier", "inference")
    web_search = _load_tier_module(_SRC_DIR / "retrieval", "web_search")
    extract_page_text = _load_tier_module(_SRC_DIR / "retrieval", "extract_page_text")
    cascade_module = _load_tier_module(_SRC_DIR / "pipeline", "cascade_pipeline")
    return tang0_inference, web_search, extract_page_text, cascade_module


# ============================================================
# BƯỚC 1 — ĐỌC FILE (.txt / .docx / .pdf)
# ============================================================

def read_file(file_path: str) -> str:
    """Đọc nội dung văn bản từ file .txt, .docx hoặc .pdf."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    elif suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise ImportError("Chưa cài python-docx. Chạy: pip install python-docx")
        document = docx.Document(str(file_path))
        return "\n".join(p.text for p in document.paragraphs)

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Chưa cài pdfplumber. Chạy: pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {suffix}. Chỉ hỗ trợ .txt, .docx, .pdf")


# ============================================================
# BƯỚC 2 — TÁCH CÂU
# ============================================================

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ0-9\"'])")


def split_sentences(text: str, min_length: int = 15) -> list:
    """
    Tách văn bản thành danh sách câu bằng regex dựa trên dấu câu kết
    thúc (., !, ?, …) theo sau bởi khoảng trắng và chữ hoa/số — hỗ
    trợ cả tiếng Việt (có dấu) và tiếng Anh.

    LƯU Ý: đây là bộ tách câu ĐƠN GIẢN, có thể tách sai với câu chứa
    viết tắt ("T.S. Nguyễn Văn A", "Mr. Smith") hoặc số thập phân
    ("giá 3.5 triệu"). Nếu cần độ chính xác cao hơn, cân nhắc dùng
    thư viện chuyên dụng (underthesea cho tiếng Việt, nltk.sent_tokenize
    cho tiếng Anh).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_sentences = _SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= min_length]

    return sentences


# ============================================================
# BƯỚC 3 — TÌM CÂU KHỚP NHẤT TRONG 1 TRANG WEB (TF-IDF local, nhẹ)
# ============================================================

def find_best_matching_sentence(query_sentence: str, page_text: str) -> str:
    """
    Trong nội dung 1 trang web đã trích xuất, tìm CÂU nào khớp nhất
    với query_sentence.

    LƯU Ý: đây là TF-IDF fit RIÊNG trên các câu của TRANG NÀY (dùng
    min_df=1 vì corpus nhỏ, chỉ 1 trang) — khác hoàn toàn với Tầng 1
    (dùng vectorizer dùng chung, fit sẵn trên corpus tham chiếu lớn).
    Đây là bài toán khác: "tìm câu ứng viên trong phạm vi 1 trang",
    không phải "lọc 1 cặp câu cho sẵn".
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    page_sentences = split_sentences(page_text)
    if not page_sentences:
        return ""

    try:
        vectorizer = TfidfVectorizer(min_df=1)
        all_texts = [query_sentence] + page_sentences
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # Trường hợp hiếm: toàn bộ câu đều là stopword/rỗng sau xử lý
        return page_sentences[0]

    query_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
    best_idx = int(similarities.argmax())

    return page_sentences[best_idx]


# ============================================================
# BƯỚC 4 — XUẤT FILE HTML HIGHLIGHT
# ============================================================

def export_highlighted_html(process_result: dict, output_path: str, original_sentences: list = None):
    """
    Xuất file HTML, highlight các câu nghi ngờ theo mức độ (đỏ/cam/
    vàng theo ngưỡng xác suất, đúng thiết kế đã thống nhất).

    Tách riêng khỏi class SingleFilePipeline để có thể TEST ĐỘC LẬP
    bằng dữ liệu giả lập, không cần chạy qua toàn bộ model AI thật.
    """
    results_by_sentence = {r["sentence"]: r for r in process_result["results"]}

    if original_sentences is None:
        original_text = read_file(process_result["file_path"])
        original_sentences = split_sentences(original_text)

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.8; }",
        ".high { background-color: #ffcccc; }",
        ".medium { background-color: #ffe0b3; }",
        ".low { background-color: #fff9cc; }",
        ".tooltip { font-size: 0.85em; color: #555; display: block; margin: 4px 0 12px 0; }",
        "</style></head><body>",
        f"<h2>Kết quả kiểm tra: {Path(process_result['file_path']).name}</h2>",
        f"<p>Tổng số câu: {process_result['total_sentences']} | "
        f"Số câu nghi ngờ: {process_result['suspicious_sentences']}</p><hr>",
    ]

    for sentence in original_sentences:
        result = results_by_sentence.get(sentence)

        if result is None or result.get("probability") is None:
            html_parts.append(f"<p>{sentence}</p>")
            continue

        prob = result["probability"]
        css_class = "high" if prob >= 0.85 else "medium" if prob >= 0.6 else "low"

        tooltip = (
            f"Nguồn: {result.get('source_url', 'N/A')} | "
            f"Xác suất: {prob:.1%} | Nhãn: {result['label']}"
        )
        if result.get("explanation"):
            tooltip += f" | Giải thích: {result['explanation']}"

        html_parts.append(f"<p><span class='{css_class}'>{sentence}</span>")
        html_parts.append(f"<span class='tooltip'>{tooltip}</span></p>")

    html_parts.append("</body></html>")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")

    logger.info(f"Đã xuất file highlight: {output_path}")
    return output_path


# ============================================================
# PIPELINE CHÍNH (cần model AI thật — torch/sentence-transformers)
# ============================================================

class SingleFilePipeline:
    def __init__(
        self,
        tang0_checkpoint_dir: str = str(TANG0_CHECKPOINT_DIR),
        max_search_results: int = WEB_SEARCH_MAX_RESULTS,
        enable_tang5: bool = False,
    ):
        self.tang0_checkpoint_dir = tang0_checkpoint_dir
        self.max_search_results = max_search_results

        check_required_keys(require_web_search=True, require_tang5=enable_tang5)
        ensure_all_dirs()

        (self._tang0_inference, self._web_search,
         self._extract_page_text, self._cascade_module) = _load_modules()

        self.cascade_pipeline = self._cascade_module.CaTEXPipeline(
            enable_tang5=enable_tang5,
            verbose=False,
        )

    def process_file(self, file_path: str) -> dict:
        """
        Xử lý toàn bộ 1 file, trả về kết quả chi tiết cho từng câu
        nghi ngờ kèm thống kê tổng quan.
        """
        logger.info(f"Đang đọc file: {file_path}")
        text = read_file(file_path)

        sentences = split_sentences(text)
        logger.info(f"Tách được {len(sentences)} câu")

        results = []

        for i, sentence in enumerate(sentences):
            logger.info(f"[{i + 1}/{len(sentences)}] Đang xử lý: {sentence[:60]}...")

            tang0_result = self._tang0_inference.predict_sentence(
                sentence, checkpoint_dir=self.tang0_checkpoint_dir
            )

            if tang0_result["decision"] == "skip":
                continue  # Không đáng nghi -> bỏ qua, KHÔNG tốn Web Search

            search_results = self._web_search.search_web(sentence, max_results=self.max_search_results)

            if not search_results:
                results.append({
                    "sentence": sentence, "tang0_probability": tang0_result["probability"],
                    "source_url": None, "label": "không tìm thấy nguồn", "probability": None,
                })
                continue

            best_result = None
            for search_item in search_results:
                page_result = self._extract_page_text.fetch_and_extract_text(search_item["url"])
                if not page_result["success"]:
                    continue

                candidate_sentence = find_best_matching_sentence(sentence, page_result["text"])
                if not candidate_sentence:
                    continue

                pipeline_result = self.cascade_pipeline.predict(sentence, candidate_sentence)

                if best_result is None or pipeline_result["probability"] > best_result["probability"]:
                    best_result = {
                        "sentence": sentence, "tang0_probability": tang0_result["probability"],
                        "source_url": search_item["url"], "matched_source_sentence": candidate_sentence,
                        "label": pipeline_result["label"], "probability": pipeline_result["probability"],
                        "explanation": pipeline_result.get("explanation"),
                    }

            results.append(best_result or {
                "sentence": sentence, "tang0_probability": tang0_result["probability"],
                "source_url": None, "label": "không trích xuất được nội dung nguồn", "probability": None,
            })

        return {
            "file_path": file_path,
            "total_sentences": len(sentences),
            "suspicious_sentences": len(results),
            "results": results,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File cần kiểm tra (.txt/.docx/.pdf)")
    parser.add_argument("--output", default="output_highlighted.html")
    parser.add_argument("--enable-tang5", action="store_true")
    args = parser.parse_args()

    pipeline = SingleFilePipeline(enable_tang5=args.enable_tang5)
    result = pipeline.process_file(args.input)
    export_highlighted_html(result, args.output)
