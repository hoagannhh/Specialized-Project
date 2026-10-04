"""
Tầng 5 — Gọi API LLM và parse kết quả
========================================

Hỗ trợ 2 nhà cung cấp phổ biến: OpenAI (GPT) và Anthropic (Claude).
Chọn 1 trong 2 tùy theo API key bạn có sẵn.

QUAN TRỌNG: LLM đôi khi trả về JSON kèm theo text thừa (markdown code
fence ```json ... ```, hoặc vài câu giải thích trước/sau) dù đã yêu
cầu rõ trong prompt — hàm parse ở đây xử lý các trường hợp này một
cách an toàn, không crash chương trình nếu LLM "làm sai yêu cầu".
"""

import json
import os
import re

from prompts import build_judge_prompt


def _extract_json_from_text(text: str) -> dict:
    """
    Trích xuất JSON từ text trả về của LLM, xử lý các trường hợp
    thường gặp:
        1. JSON thuần túy (trường hợp lý tưởng)
        2. JSON bọc trong ```json ... ``` (markdown code fence)
        3. JSON có thêm câu giải thích trước/sau

    Nếu không parse được, trả về dict mặc định với cờ "parse_error"
    thay vì crash toàn bộ pipeline.
    """
    text = text.strip()

    # Trường hợp 1: thử parse trực tiếp
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Trường hợp 2: tìm trong code fence ```json ... ``` hoặc ``` ... ```
    code_fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_fence_match:
        try:
            return json.loads(code_fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Trường hợp 3: tìm khối {...} đầu tiên trong text (kể cả có chữ xung quanh)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Không parse được -> trả về mặc định an toàn, KHÔNG crash
    return {
        "is_paraphrase": None,
        "confidence": None,
        "changed_phrases": [],
        "explanation": None,
        "parse_error": True,
        "raw_response": text[:500],  # giữ lại 1 phần để debug sau
    }


def call_openai(prompt: str, model: str = "gpt-4o-mini", api_key: str = None) -> str:
    """Gọi OpenAI API, trả về text thô (chưa parse JSON)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Chưa cài thư viện openai. Chạy: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Thiếu OpenAI API key. Truyền qua tham số api_key hoặc đặt "
            "biến môi trường OPENAI_API_KEY."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Giảm ngẫu nhiên, ưu tiên nhất quán cho tác vụ đánh giá
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-5", api_key: str = None) -> str:
    """Gọi Anthropic API, trả về text thô (chưa parse JSON)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Chưa cài thư viện anthropic. Chạy: pip install anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Thiếu Anthropic API key. Truyền qua tham số api_key hoặc đặt "
            "biến môi trường ANTHROPIC_API_KEY."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def tang5_judge(
    sentence_a: str,
    sentence_b: str,
    model_label: str,
    model_probability: float,
    provider: str = "openai",
    api_key: str = None,
) -> dict:
    """
    Hàm CHÍNH của Tầng 5 — gọi LLM, parse kết quả an toàn.

    Parameters
    ----------
    provider : "openai" hoặc "anthropic"
    """
    prompt = build_judge_prompt(sentence_a, sentence_b, model_label, model_probability)

    if provider == "openai":
        raw_text = call_openai(prompt, api_key=api_key)
    elif provider == "anthropic":
        raw_text = call_anthropic(prompt, api_key=api_key)
    else:
        raise ValueError(f"provider không hợp lệ: {provider} (chỉ nhận 'openai' hoặc 'anthropic')")

    result = _extract_json_from_text(raw_text)

    if result.get("parse_error"):
        print(
            "CẢNH BÁO: không parse được JSON từ phản hồi của LLM. "
            "Xem 'raw_response' trong kết quả để debug. Hệ thống vẫn "
            "tiếp tục chạy (không crash), dùng nhãn từ Tầng 4 làm "
            "kết quả cuối, không có giải thích chi tiết."
        )

    return result


if __name__ == "__main__":
    # Test parse JSON với vài trường hợp "lộn xộn" thường gặp từ LLM thật
    test_cases = [
        '{"is_paraphrase": true, "confidence": 0.9, "changed_phrases": [], "explanation": "test"}',
        '```json\n{"is_paraphrase": true, "confidence": 0.9, "changed_phrases": [], "explanation": "test"}\n```',
        'Đây là kết quả phân tích:\n{"is_paraphrase": false, "confidence": 0.3, "changed_phrases": [], "explanation": "test"}\nHy vọng hữu ích!',
        'Xin lỗi, tôi không thể xử lý yêu cầu này.',  # trường hợp hoàn toàn không có JSON
    ]

    for i, raw in enumerate(test_cases, 1):
        parsed = _extract_json_from_text(raw)
        print(f"--- Test case {i} ---")
        print(f"Input : {raw[:80]}...")
        print(f"Output: {parsed}")
        print()
