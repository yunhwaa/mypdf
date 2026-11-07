"""PDF 텍스트 추출 유틸리티."""

from __future__ import annotations

import hashlib
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

STATUS_OK = "ok"
STATUS_ENCRYPTED = "encrypted"
STATUS_NO_TEXT = "no_text"
STATUS_ERROR = "error"


# Streamlit Cloud 메모리(약 1GB)를 고려해 캐시 개수와 보관 시간을 제한한다.
@st.cache_data(show_spinner=False, max_entries=20, ttl=3600)
def extract_pdf(file_bytes: bytes, filename: str) -> dict:
    """PDF 바이트에서 페이지별 텍스트를 추출한다.

    캐시 키는 파일 내용(bytes)과 파일명이므로, 이름이 같아도 내용이 다르면 다시 추출한다.
    파일은 디스크에 저장하지 않고 메모리에서만 처리한다.
    """
    result = {
        "id": hashlib.sha1(file_bytes).hexdigest(),
        "name": filename,
        "pages": 0,
        "chars": 0,
        "text": "",
        "page_texts": [],  # "[파일명 - p.N]\n본문" 형태의 페이지별 텍스트
        "status": STATUS_OK,
        "message": "",
    }

    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            # 빈 암호로 열리는 PDF(권한 제한만 걸린 경우)는 허용한다.
            try:
                opened = reader.decrypt("")
            except Exception:
                opened = 0
            if not opened:
                result["status"] = STATUS_ENCRYPTED
                result["message"] = "암호로 보호된 PDF는 지원하지 않습니다."
                return result

        result["pages"] = len(reader.pages)
        page_texts = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                page_texts.append(f"[{filename} - p.{i}]\n{text}")
    except Exception as e:  # 손상된 파일, 지원하지 않는 암호화 방식 등 어떤 경우에도 앱이 멈추지 않게
        result["status"] = STATUS_ERROR
        detail = str(e) if isinstance(e, (PdfReadError, ValueError, OSError)) else type(e).__name__
        result["message"] = f"PDF를 읽을 수 없습니다: {detail}"
        return result

    full_text = "\n\n".join(page_texts)
    result["page_texts"] = page_texts
    result["text"] = full_text
    result["chars"] = len(full_text)

    if not page_texts:
        result["status"] = STATUS_NO_TEXT
        result["message"] = "텍스트를 추출할 수 없는 PDF입니다 (이미지 기반 스캔본일 수 있음)."

    return result
