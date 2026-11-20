"""
Tầng 5 — Gọi API LLM và parse kết quả
========================================

Hỗ trợ OpenAI (GPT), Anthropic (Claude) và Google Gemini.
Chọn provider theo API key được cấu hình trong môi trường.

QUAN TRỌNG: LLM đôi khi trả về JSON kèm theo text thừa (markdown code
fence ```json ... ```, hoặc vài câu giải thích trước/sau) dù đã yêu
cầu rõ trong prompt — hàm parse ở đây xử lý các trường hợp này một
cách an toàn, không crash chương trình nếu LLM "làm sai yêu cầu".
"""

import json
import os
import re
from time import monotonic
from threading import Lock

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.tang5_llm_judge.prompts import build_judge_prompt
from src.utils.config import configured_llm_keys

_gemini_key_lock = Lock()
_gemini_active_key = {}
_gemini_retry_at = {}


class GeminiQuotaExhausted(Exception):
    """Mọi key đã cấu hình cho model này đều đang chờ quota."""


def _gemini_quota_delay(error):
    """Đọc số giây chờ từ Gemini, mặc định một phút nếu thiếu thông tin."""
    headers = getattr(error.response, 'headers', {})
    hints = [headers.get('retry-after')]
    payload = error.details if isinstance(error.details, dict) else {}
    error_body = payload.get('error', payload)
    details = error_body.get('details') if isinstance(error_body, dict) else None
    for detail in details or []:
        if isinstance(detail, dict) and detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
            hints.append(detail.get('retryDelay'))
    seconds = [float(str(hint).removesuffix('s')) for hint in hints
               if hint is not None and re.fullmatch(r'\d+(?:\.\d+)?s?', str(hint))]
    return max(1.0, max(seconds, default=60.0))


def _extract_json_from_text(text: str) -> dict:
    """Reject valid JSON with the wrong schema, not only malformed JSON."""
    if not isinstance(text, str):
        text = ''
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        phrases = value.get('changed_phrases')
        explanation = value.get('explanation')
        if not isinstance(explanation, str) or not explanation.strip() or not isinstance(phrases, list):
            continue
        if any(not isinstance(p, dict) or any(not isinstance(p.get(k), str) or not p[k].strip()
               for k in ('original','changed_to')) for p in phrases):
            continue
        # Discard any model-invented label/confidence fields.
        return {'changed_phrases': [{'original':p['original'], 'changed_to':p['changed_to']} for p in phrases],
                'explanation': explanation, 'parse_error': False}
    return {'changed_phrases': [], 'explanation': None, 'parse_error': True}


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


def call_gemini(prompt: str, model: str = None, api_key: str = None) -> str:
    """Gọi Google Gemini API, trả về text thô (chưa parse JSON)."""
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        raise ImportError("Chưa cài thư viện Google GenAI. Chạy: pip install google-genai")

    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    request_config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")

    def generate(key):
        client = genai.Client(api_key=key)
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=request_config)
            return response.text or ""
        finally:
            close = getattr(client, "close", None)
            if close:
                close()

    if api_key:
        return generate(api_key)
    keys = configured_llm_keys("gemini")
    if not keys:
        raise ValueError("Chưa cấu hình GEMINI_API_KEYS hoặc GEMINI_API_KEY cho tầng 5.")

    attempted = set()
    while len(attempted) < len(keys):
        # Chỉ khóa lúc chọn key. Các yêu cầu mạng có thể chạy đồng thời.
        with _gemini_key_lock:
            now = monotonic()
            available = [key for key in keys if key not in attempted
                         and _gemini_retry_at.get((model, key), 0) <= now]
            if not available:
                break
            active = _gemini_active_key.get(model)
            key = active if active in available else available[0]
            _gemini_active_key[model] = key
        attempted.add(key)
        try:
            return generate(key)
        except errors.APIError as exc:
            if exc.code != 429:
                raise
            retry_at = monotonic() + _gemini_quota_delay(exc)
            with _gemini_key_lock:
                slot = (model, key)
                _gemini_retry_at[slot] = max(_gemini_retry_at.get(slot, 0), retry_at)
    raise GeminiQuotaExhausted("Các key Gemini đã cấu hình đều đang bị giới hạn. Vui lòng thử lại sau.")


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
    provider : "openai", "anthropic" hoặc "gemini"
    """
    # Compatibility: the document pipeline passes suspect first, source second.
    suspect_sentence, source_sentence = sentence_a, sentence_b
    prompt = build_judge_prompt(source_sentence=source_sentence, suspect_sentence=suspect_sentence,
                                model_label=model_label, model_probability=model_probability)

    if provider == "openai":
        raw_text = call_openai(prompt, api_key=api_key)
    elif provider == "anthropic":
        raw_text = call_anthropic(prompt, api_key=api_key)
    elif provider == "gemini":
        raw_text = call_gemini(prompt, api_key=api_key)
    else:
        raise ValueError(
            f"provider không hợp lệ: {provider} "
            "(chỉ nhận 'openai', 'anthropic' hoặc 'gemini')"
        )

    result = _extract_json_from_text(raw_text)

    if not result['parse_error']:
        for phrase in result['changed_phrases']:
            if phrase['original'] not in source_sentence or phrase['changed_to'] not in suspect_sentence:
                result = {'changed_phrases': [], 'explanation': None, 'parse_error': True,
                          'error': 'unsupported_evidence_span'}
                break

    if result.get("parse_error"):
        print(
            "CẢNH BÁO: không parse được JSON từ phản hồi của LLM. "
            "Phản hồi sai cấu trúc hoặc thiếu bằng chứng hợp lệ. Hệ thống vẫn "
            "tiếp tục chạy (không crash), giữ nhãn định lượng của pipeline làm "
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
