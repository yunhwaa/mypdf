"""OpenAI 설정 로딩, API 호출, 분석·Q&A 로직."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator

import openai
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"
MAX_RETRIES = 3  # 429 재시도 횟수
MAX_HISTORY_MESSAGES = 10  # Q&A에 포함할 최근 대화 수
QA_CHUNK_TOKENS = 1_500  # Q&A 관련 구간 검색용 청크 크기
RESPONSE_RESERVE_TOKENS = 4_000  # 프롬프트·응답을 위해 남겨둘 여유분


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def get_config(key: str, default: str | None = None) -> str | None:
    """설정값을 st.secrets(Streamlit Cloud) → 환경변수/.env(로컬) 순서로 읽는다."""
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value:
                return str(value)
    except Exception:
        # 로컬에 .streamlit/secrets.toml이 없으면 예외가 발생하므로 무시한다.
        pass
    value = os.getenv(key)
    return value if value else default


def _int_config(key: str, default: int) -> int:
    try:
        return max(1_000, int(get_config(key) or default))
    except ValueError:
        return default


def get_api_key() -> str | None:
    return get_config("OPENAI_API_KEY")


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


# 한 번에 보낼 수 있는 최대 입력 토큰 수. 초과하면 Map-Reduce로 처리한다.
MAX_INPUT_TOKENS = _int_config("MAX_INPUT_TOKENS", 100_000)
# Map 단계에서 한 번에 요약할 청크 크기
MAP_CHUNK_TOKENS = min(_int_config("MAP_CHUNK_TOKENS", 20_000), MAX_INPUT_TOKENS)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    # 재시도는 _create()에서 직접 제어하므로 SDK 자동 재시도는 끈다.
    return OpenAI(api_key=api_key, max_retries=0, timeout=180)


# ---------------------------------------------------------------------------
# 오류 처리
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 오류 메시지를 담는 예외."""


def friendly_error(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. OPENAI_API_KEY를 확인하세요."
    if isinstance(e, openai.RateLimitError):
        if getattr(e, "code", None) == "insufficient_quota":
            return "OpenAI 크레딧(사용 한도)이 부족합니다. 결제 설정을 확인하세요."
        return "요청이 많습니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, openai.NotFoundError):
        return "모델을 찾을 수 없습니다. OPENAI_MODEL 값을 확인하세요."
    if isinstance(e, openai.PermissionDeniedError):
        return "이 API 키로는 해당 모델에 접근할 권한이 없습니다."
    if isinstance(e, openai.APITimeoutError):
        return "OpenAI 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
    if isinstance(e, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하세요."
    if isinstance(e, openai.BadRequestError):
        if getattr(e, "code", None) == "context_length_exceeded":
            return "문서가 모델의 입력 한도를 초과했습니다. MAX_INPUT_TOKENS 값을 낮춰보세요."
        return f"잘못된 요청입니다: {getattr(e, 'message', e)}"
    if isinstance(e, openai.InternalServerError):
        return "OpenAI 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    return f"OpenAI API 오류: {e}"


def is_mock() -> bool:
    """테스트 모드: MOCK_LLM=1이면 OpenAI를 호출하지 않고 고정된 가짜 응답을 돌려준다.

    API 비용 없이 E2E 테스트를 반복하기 위한 용도이며 기본값은 꺼져 있다.
    """
    return get_config("MOCK_LLM", "0") == "1"


def check_connection(api_key: str, model: str) -> tuple[bool, str]:
    """모델 조회 API로 키와 모델명이 유효한지 확인한다. (토큰을 소모하지 않음)"""
    if is_mock():
        return True, f"연결 성공 (테스트 모드): `{model}`"
    try:
        get_client(api_key).models.retrieve(model)
        return True, f"연결 성공: `{model}` 모델을 사용할 수 있습니다."
    except openai.NotFoundError:
        return False, f"`{model}` 모델을 찾을 수 없습니다. OPENAI_MODEL 값을 확인하세요."
    except openai.OpenAIError as e:
        return False, friendly_error(e)


# ---------------------------------------------------------------------------
# API 호출
# ---------------------------------------------------------------------------
# 모델이 temperature 변경을 지원하지 않으면 False로 기록 (예: 추론 모델)
_temperature_supported: dict[str, bool] = {}


def temperature_supported(model: str) -> bool | None:
    """아직 확인 전이면 None."""
    return _temperature_supported.get(model)


def _mock_text(messages: list[dict], temperature: float) -> str:
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[-1]["content"]
    task = user.split("[작업]\n", 1)[-1] if "[작업]\n" in user else user
    task = task.split("[최종 작업]\n", 1)[-1] if "[최종 작업]\n" in task else task
    history = len([m for m in messages[1:-1] if m["role"] in ("user", "assistant")])
    language = "English" if "반드시 English로" in system else "한국어"
    source = "요약본" if "구간별로 요약한 것" in user else "발췌" if "관련된 부분만 발췌" in system else "원문"
    return (
        f"[MOCK 응답] 언어={language} | 이전 대화={history}개 | 입력={source} | temperature={temperature:.1f}\n\n"
        f"요청: {task.strip()[:60]}\n\n"
        "- 긍정 단어 비율은 **약 73%**로 나타났다. [mock.pdf - p.1]\n"
        "- 외부 이미지 차단 확인: ![leak](https://example.invalid/leak.png)\n"
        "- 두 번째 핵심 포인트입니다. [mock.pdf - p.2]"
    )


def _mock_response(messages: list[dict], temperature: float, stream: bool):
    from types import SimpleNamespace as NS

    text = _mock_text(messages, temperature)
    if not stream:
        return NS(choices=[NS(message=NS(content=text))])

    def chunks():
        for i in range(0, len(text), 20):
            time.sleep(0.02)
            yield NS(choices=[NS(delta=NS(content=text[i : i + 20]))])

    return chunks()


def _create(api_key: str, model: str, messages: list[dict], temperature: float, stream: bool):
    if is_mock():
        return _mock_response(messages, temperature, stream)
    client = get_client(api_key)
    kwargs: dict = {"model": model, "messages": messages, "stream": stream}
    if _temperature_supported.get(model, True):
        kwargs["temperature"] = temperature

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(**kwargs)
            if "temperature" in kwargs:
                _temperature_supported[model] = True
            return resp
        except openai.RateLimitError as e:
            if getattr(e, "code", None) == "insufficient_quota" or attempt >= MAX_RETRIES:
                raise LLMError(friendly_error(e)) from e
            time.sleep(2**attempt)  # 1, 2, 4초 지수 백오프
            attempt += 1
        except openai.BadRequestError as e:
            if "temperature" in kwargs and "temperature" in str(e):
                # 기본값만 허용하는 모델 → temperature 없이 재시도
                _temperature_supported[model] = False
                kwargs.pop("temperature")
                continue
            raise LLMError(friendly_error(e)) from e
        except openai.OpenAIError as e:
            raise LLMError(friendly_error(e)) from e


def stream_chat(api_key: str, model: str, messages: list[dict], temperature: float) -> Iterator[str]:
    """응답을 텍스트 조각 단위로 스트리밍한다. 오류는 LLMError로 변환된다."""
    stream = _create(api_key, model, messages, temperature, stream=True)
    try:
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
    except openai.OpenAIError as e:
        raise LLMError(friendly_error(e)) from e


def complete(api_key: str, model: str, messages: list[dict], temperature: float) -> str:
    resp = _create(api_key, model, messages, temperature, stream=False)
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 토큰 계산 / 청크 분할
# ---------------------------------------------------------------------------
class _CharEncoding:
    """tiktoken 인코딩 파일을 내려받지 못할 때 쓰는 근사 인코더 (글자 1개 = 토큰 1개).

    실제 토큰 수보다 크게 세므로 한도 계산이 보수적으로 동작한다.
    """

    def encode(self, text: str, disallowed_special=()) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


@st.cache_resource(show_spinner=False)
def _get_encoding(model: str):
    try:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")
    except Exception:
        # 첫 사용 시 인코딩 파일을 인터넷에서 내려받는데, 실패해도 앱이 멈추지 않게 한다.
        return _CharEncoding()


def _encode(text: str, model: str) -> list[int]:
    return _get_encoding(model).encode(text, disallowed_special=())


@st.cache_data(show_spinner=False, max_entries=200)
def count_tokens(text: str, model: str) -> int:
    return len(_encode(text, model))


def split_into_chunks(page_texts: list[str], max_tokens: int, model: str) -> list[str]:
    """페이지 경계를 기준으로 max_tokens 이하의 청크로 나눈다.

    한 페이지가 max_tokens보다 크면 토큰 단위로 잘라 여러 청크로 만든다.
    빈 청크는 만들지 않는다.
    """
    enc = _get_encoding(model)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        current, current_tokens = [], 0

    for page in page_texts:
        if not page.strip():
            continue
        tokens = enc.encode(page, disallowed_special=())
        n = len(tokens)
        if n > max_tokens:
            flush()
            marker = f"{page.split(chr(10), 1)[0]} (계속)\n" if page.startswith("[") else ""
            # 이어지는 조각 앞에 붙는 출처 표시만큼 여유를 둔다.
            step = max(1, max_tokens - len(enc.encode(marker, disallowed_special=())) - 2)
            for i in range(0, n, step):
                piece = enc.decode(tokens[i : i + step])
                chunks.append(f"{marker}{piece}" if i > 0 else piece)
            continue
        if current and current_tokens + n > max_tokens:
            flush()
        current.append(page)
        current_tokens += n
    flush()
    return chunks


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
ANALYSIS_TYPES: dict[str, str] = {
    "전체 요약": (
        "각 문서를 다음 Markdown 구조로 요약하라. 문서가 여러 개면 문서별로 작성한다.\n"
        "## 📄 <문서명>\n### 목적\n### 주요 내용\n### 결론"
    ),
    "핵심 포인트": (
        "문서에서 가장 중요한 핵심 포인트를 중요도 순으로 5~10개의 불릿으로 정리하라. "
        "각 불릿은 한두 문장으로 쓰고 끝에 출처를 표기하라."
    ),
    "문서 간 비교": (
        "업로드된 문서들을 비교 분석하라.\n"
        "1. 각 문서를 한두 문장으로 소개한다.\n"
        "2. 공통점과 차이점을 Markdown 표로 정리한다. (비교 항목 | 문서별 내용)\n"
        "3. 종합 의견을 작성한다."
    ),
    "사용자 지정": "",
}

LANGUAGES = ["한국어", "English"]


def system_prompt(language: str) -> str:
    not_found = (
        "'This cannot be found in the documents.'" if language == "English" else "'문서에서 확인할 수 없습니다'"
    )
    return (
        "너는 문서 분석 전문가다. 사용자가 업로드한 PDF 내용만을 근거로 답하라.\n"
        "- 근거가 된 내용에는 [파일명 - p.N] 형식으로 출처를 표기하라.\n"
        f"- 문서에 없는 내용은 추측하지 말고 {not_found}라고 답하라.\n"
        "- 결과는 읽기 쉬운 Markdown 형식으로 작성하라.\n"
        f"- 모든 답변은 반드시 {language}로 작성하라."
    )


def _attr(name: str) -> str:
    """파일명을 프롬프트의 태그 속성에 넣을 때 태그 구조가 깨지지 않게 이스케이프한다."""
    return name.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def format_docs(docs: list[dict]) -> str:
    return "\n\n".join(f'<document name="{_attr(d["name"])}">\n{d["text"]}\n</document>' for d in docs)


def _analysis_budget(instruction: str, model: str) -> int:
    """분석 요청에서 문서 컨텍스트에 쓸 수 있는 토큰 수 (지시문·응답 여유분 제외)."""
    return MAX_INPUT_TOKENS - RESPONSE_RESERVE_TOKENS - count_tokens(instruction, model)


# ---------------------------------------------------------------------------
# 분석 (긴 문서는 Map-Reduce)
# ---------------------------------------------------------------------------
ProgressFn = Callable[[int, int, int], None]  # (완료 수, 전체 수, 단계)


@st.cache_data(show_spinner=False, max_entries=500, ttl=3600)
def _summarize_chunk(
    _api_key: str, model: str, name: str, chunk: str, instruction: str, temperature: float
) -> str:
    """청크 하나를 요약한다 (Map 단계).

    결과를 캐시하므로, 긴 분석 도중 사용자가 다른 위젯을 눌러 실행이 중단되더라도
    다시 분석할 때 이미 요약한 청크는 API를 다시 호출하지 않는다.
    (_api_key는 밑줄로 시작해 캐시 키에서 제외된다.)
    """
    messages = [
        {"role": "system", "content": system_prompt("한국어")},
        {
            "role": "user",
            "content": (
                f"다음은 문서 '{name}'의 일부다. 최종 작업에 필요한 핵심 내용, 주요 수치, "
                "결론을 빠짐없이 간결하게 요약하라. 원문의 [파일명 - p.N] 출처 표기를 유지하라.\n\n"
                f"[최종 작업]\n{instruction}\n\n[문서 일부]\n{chunk}"
            ),
        },
    ]
    return complete(_api_key, model, messages, temperature)


def prepare_analysis_context(
    api_key: str,
    model: str,
    docs: list[dict],
    instruction: str,
    temperature: float,
    on_progress: ProgressFn | None = None,
) -> tuple[str, bool]:
    """분석에 넣을 문서 컨텍스트를 만든다. 반환: (컨텍스트, 요약본 사용 여부)

    전체 문서가 MAX_INPUT_TOKENS 이내면 원문 그대로, 초과하면 청크별 요약(Map)을 만든다.
    요약본도 한도를 넘으면 요약본을 다시 요약한다 (최대 3단계).
    """
    budget = _analysis_budget(instruction, model)
    full = format_docs(docs)
    if count_tokens(full, model) <= budget:
        return full, False

    units: list[tuple[str, list[str]]] = [(d["name"], d["page_texts"]) for d in docs]
    for level in range(1, 4):
        chunks = [
            (name, chunk) for name, pages in units for chunk in split_into_chunks(pages, MAP_CHUNK_TOKENS, model)
        ]
        summaries: list[tuple[str, str]] = []
        for i, (name, chunk) in enumerate(chunks):
            if on_progress:
                on_progress(i, len(chunks), level)
            summaries.append((name, _summarize_chunk(api_key, model, name, chunk, instruction, temperature)))
        if on_progress:
            on_progress(len(chunks), len(chunks), level)

        context = "\n\n".join(
            f'<document_summary name="{_attr(name)}" part="{i + 1}">\n{s}\n</document_summary>'
            for i, (name, s) in enumerate(summaries)
        )
        if count_tokens(context, model) <= budget:
            return context, True

        grouped: dict[str, list[str]] = {}
        for name, s in summaries:
            grouped.setdefault(name, []).append(s)
        units = list(grouped.items())

    raise LLMError("문서가 너무 길어 처리할 수 없습니다. 파일 수를 줄여 다시 시도해주세요.")


def build_analysis_messages(context: str, instruction: str, language: str, summarized: bool) -> list[dict]:
    note = "(아래 내용은 긴 문서를 구간별로 요약한 것이다.)\n" if summarized else ""
    return [
        {"role": "system", "content": system_prompt(language)},
        {"role": "user", "content": f"{note}[문서]\n{context}\n\n[작업]\n{instruction}"},
    ]


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
def _bigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for word in re.findall(r"\w+", text.lower()):
        if len(word) >= 2:
            grams.update(word[i : i + 2] for i in range(len(word) - 1))
    return grams


def select_relevant_context(docs: list[dict], query: str, budget: int, model: str) -> str:
    """질문과 겹치는 단어(2글자 단위)가 많은 구간을 우선 골라 budget 토큰 이내로 만든다."""
    query_grams = _bigrams(query)
    candidates = []  # (점수, 원래 순서, 문서명, 청크, 토큰 수)
    order = 0
    for d in docs:
        for chunk in split_into_chunks(d["page_texts"], QA_CHUNK_TOKENS, model):
            score = len(query_grams & _bigrams(chunk))
            candidates.append((score, order, d["name"], chunk, count_tokens(chunk, model)))
            order += 1

    selected, used = [], 0
    for cand in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if used + cand[4] > budget:
            continue
        selected.append(cand)
        used += cand[4]

    selected.sort(key=lambda c: c[1])
    return "\n\n".join(
        f'<document_excerpt name="{_attr(c[2])}">\n{c[3]}\n</document_excerpt>' for c in selected
    )


def build_qa_messages(
    docs: list[dict], history: list[dict], question: str, language: str, model: str
) -> list[dict]:
    fixed = count_tokens(system_prompt(language), model) + count_tokens(question, model) + RESPONSE_RESERVE_TOKENS
    # 문서 발췌에 최소 QA_CHUNK_TOKENS는 남도록, 넘치면 오래된 대화부터 뺀다.
    recent = history[-MAX_HISTORY_MESSAGES:]
    while recent and fixed + sum(count_tokens(m["content"], model) for m in recent) > MAX_INPUT_TOKENS - QA_CHUNK_TOKENS:
        recent = recent[2:] if len(recent) >= 2 else []
    history_tokens = sum(count_tokens(m["content"], model) for m in recent)
    # 태그 등 형식 문자열 몫으로 5% 여유를 둔다.
    budget = max(QA_CHUNK_TOKENS, int((MAX_INPUT_TOKENS - fixed - history_tokens) * 0.95))

    full = format_docs(docs)
    if count_tokens(full, model) <= budget:
        context, note = full, ""
    else:
        recent_questions = " ".join(m["content"] for m in recent if m["role"] == "user")
        context = select_relevant_context(docs, f"{question} {recent_questions}", budget, model)
        note = "(문서가 길어 질문과 관련된 부분만 발췌했다.)\n"

    system = f"{system_prompt(language)}\n\n아래는 사용자가 업로드한 문서 내용이다. {note}\n{context}"
    return [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}]
