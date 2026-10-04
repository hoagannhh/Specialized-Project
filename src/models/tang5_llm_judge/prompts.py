"""
Tầng 5 — Prompt template cho LLM-as-a-Judge
==============================================

Vai trò: (1) xác nhận lại nhãn từ Tầng 4, (2) chỉ ra từ/cụm từ cụ
thể đã bị thay đổi, (3) sinh giải thích bằng ngôn ngữ tự nhiên —
đúng như đã thiết kế, lấy cảm hứng từ hướng "interpretability" của
Quidwai et al. (2023).

QUAN TRỌNG: prompt yêu cầu LLM trả về JSON THUẦN TÚY, không kèm
markdown/text thừa — để parse tự động, tránh phải xử lý chuỗi lộn
xộn ở call_api.py.
"""

JUDGE_PROMPT_TEMPLATE = """Bạn là một chuyên gia phân tích ngôn ngữ, nhiệm vụ của bạn là đánh giá xem CÂU B có phải là bản diễn đạt lại (paraphrase) bằng AI của CÂU A hay không.

CÂU A (nghi là nguồn gốc):
{sentence_a}

CÂU B (nghi ngờ đạo văn):
{sentence_b}

THÔNG TIN THAM KHẢO (từ mô hình phân loại tự động, chỉ mang tính gợi ý, bạn cần tự đánh giá độc lập):
- Nhãn mô hình dự đoán: {model_label}
- Độ tin cậy của mô hình: {model_probability:.1%}

Hãy phân tích và trả lời DUY NHẤT bằng JSON theo đúng định dạng sau, không thêm bất kỳ văn bản nào khác ngoài JSON:

{{
  "is_paraphrase": true hoặc false,
  "confidence": một số từ 0 đến 1,
  "changed_phrases": [
    {{"original": "cụm từ gốc trong câu A", "changed_to": "cụm từ tương ứng trong câu B"}}
  ],
  "explanation": "giải thích ngắn gọn (2-3 câu) bằng tiếng Việt, nêu rõ vì sao kết luận như vậy"
}}

Nếu "changed_phrases" không có gì đáng chú ý (ví dụ 2 câu không liên quan), để mảng rỗng []."""


def build_judge_prompt(
    sentence_a: str,
    sentence_b: str,
    model_label: str,
    model_probability: float,
) -> str:
    """
    Điền thông tin vào template, trả về prompt hoàn chỉnh sẵn sàng
    gửi cho LLM.
    """
    return JUDGE_PROMPT_TEMPLATE.format(
        sentence_a=str(sentence_a),
        sentence_b=str(sentence_b),
        model_label=str(model_label),
        model_probability=float(model_probability),
    )


if __name__ == "__main__":
    prompt = build_judge_prompt(
        sentence_a="The economy grew rapidly last year.",
        sentence_b="Last year, the economy expanded at a fast pace.",
        model_label="đạo văn (AI-paraphrase)",
        model_probability=0.91,
    )
    print(prompt)
