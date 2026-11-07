"""PDF 분석기 — Streamlit 메인 앱."""

import hmac

import streamlit as st

from llm import (
    ANALYSIS_TYPES,
    LANGUAGES,
    MAX_INPUT_TOKENS,
    LLMError,
    build_analysis_messages,
    build_qa_messages,
    check_connection,
    count_tokens,
    get_api_key,
    get_config,
    get_model,
    is_mock,
    prepare_analysis_context,
    stream_chat,
    temperature_supported,
)
from md_utils import neutralize_image_stream, prepare_markdown
from pdf_utils import STATUS_ENCRYPTED, STATUS_NO_TEXT, STATUS_OK, extract_pdf

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

USER_AVATAR = "🙋"
AI_AVATAR = "🤖"
ANALYSIS_ICONS = {"전체 요약": "📝", "핵심 포인트": "🎯", "문서 간 비교": "⚖️", "사용자 지정": "✏️"}
EXAMPLE_QUESTIONS = [
    "이 문서의 핵심 결론은 뭐야?",
    "주요 수치를 표로 정리해줘",
    "이 문서에서 주의해야 할 점은?",
]
STATUS_LABELS = {
    STATUS_OK: "✅ 분석 가능",
    STATUS_ENCRYPTED: "⚠️ 암호화됨",
    STATUS_NO_TEXT: "⚠️ 텍스트 없음",
}

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------
DEFAULT_STATE = {
    "docs": [],  # 분석 가능한 문서 목록 (extract_pdf 결과)
    "doc_sig": (),  # 현재 문서 구성 (바뀌면 결과 초기화)
    "messages": [],  # Q&A 대화 이력 [{"role", "content"}]
    "analysis": None,  # 마지막 분석 결과 {"type", "text"}
    "pending_question": None,  # 예시 질문 버튼으로 보낸 질문
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_question = None


def ask_example(question: str):
    st.session_state.pending_question = question


def show_markdown(text: str):
    st.markdown(prepare_markdown(text))


def stream_markdown(first, stream) -> str:
    """응답을 실시간으로 출력한 뒤, 보정한 최종본(외부 이미지 제거·한글 굵게 보정)으로 다시 그린다."""
    raw: list[str] = []

    def full():
        if first is not None:
            raw.append(first)
            yield first
        for piece in stream:
            raw.append(piece)
            yield piece

    placeholder = st.empty()
    with placeholder.container():
        st.write_stream(neutralize_image_stream(full()))
    text = "".join(raw)
    placeholder.markdown(prepare_markdown(text))
    return text


# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("📄 PDF 분석기")
st.caption("PDF를 업로드하면 AI가 요약·분석하고 질문에 답해드립니다.")

api_key = get_api_key()
model = get_model()

if not api_key:
    st.error(
        "**OpenAI API 키가 설정되지 않았습니다.**\n\n"
        "- **로컬 실행**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...` 를 추가하세요.\n"
        "- **Streamlit Cloud**: App settings → Secrets 에 `OPENAI_API_KEY = \"sk-...\"` 를 추가하세요.",
        icon="🔑",
    )
    st.stop()

# 공개 배포 시 API 비용 남용을 막기 위한 선택적 비밀번호 (APP_PASSWORD를 설정한 경우에만)
app_password = get_config("APP_PASSWORD")
if app_password and not st.session_state.get("authenticated"):
    with st.form("login"):
        st.markdown("🔒 이 앱은 비밀번호로 보호되어 있습니다.")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("입장", type="primary")
    if submitted:
        if hmac.compare_digest(password.encode(), app_password.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")
    st.stop()

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    st.markdown(f"**모델** &nbsp;`{model}`")
    if is_mock():
        st.warning("테스트 모드 (MOCK_LLM=1): 실제 AI를 호출하지 않습니다.", icon="🧪")

    temp_locked = temperature_supported(model) is False
    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.3,
        0.1,
        disabled=temp_locked,
        help="낮을수록 일관되고, 높을수록 다양한 답변을 생성합니다.",
    )
    if temp_locked:
        st.caption("ℹ️ 이 모델은 Temperature 변경을 지원하지 않아 기본값으로 동작합니다.")

    language = st.radio("출력 언어", LANGUAGES, horizontal=True)

    st.divider()
    st.button("🗑️ 대화 초기화", on_click=clear_chat, width="stretch")
    if st.button("🔌 OpenAI 연결 테스트", width="stretch"):
        with st.spinner("연결 확인 중…"):
            ok, msg = check_connection(api_key, model)
        (st.success if ok else st.error)(msg)

    st.divider()
    with st.expander("📖 사용 방법", expanded=False):
        st.markdown(
            "1. 분석할 **PDF 파일을 업로드**합니다. (여러 개 가능)\n"
            "2. **📊 분석** 탭에서 분석 유형을 고르고 **분석 시작**을 누릅니다.\n"
            "3. **💬 Q&A 채팅** 탭에서 문서 내용을 질문합니다.\n"
            "4. 분석 결과는 Markdown 파일로 내려받을 수 있습니다.\n\n"
            "※ 스캔 이미지 PDF와 암호화된 PDF는 지원하지 않습니다."
        )

# ---------------------------------------------------------------------------
# PDF 업로드 및 텍스트 추출
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    accept_multiple_files=True,
    help="파일을 끌어다 놓거나 클릭해서 선택하세요. 여러 개를 한 번에 올릴 수 있습니다.",
)

docs = []
rows = []
if uploaded_files:
    with st.status("텍스트 추출 중…", expanded=False) as status:
        results = []
        for f in uploaded_files:
            status.update(label=f"텍스트 추출 중… ({f.name})")
            results.append(extract_pdf(f.getvalue(), f.name))
        status.update(label="텍스트 추출 완료", state="complete")

    seen = set()
    for r in results:
        tokens = None
        if r["status"] != STATUS_OK:
            label, note = STATUS_LABELS.get(r["status"], "❌ 오류"), r["message"]
        elif r["id"] in seen:
            label, note = "⚠️ 중복", "이미 업로드된 파일과 내용이 같아 제외했습니다."
        else:
            seen.add(r["id"])
            docs.append(r)
            tokens = count_tokens(r["text"], model)
            label, note = STATUS_LABELS[STATUS_OK], ""
        rows.append(
            {
                "상태": label,
                "파일명": r["name"],
                "페이지": r["pages"],
                "글자 수": r["chars"],
                "예상 토큰": tokens,
                "비고": note,
            }
        )

# 업로드된 문서 구성이 바뀌면 이전 분석 결과와 대화를 초기화한다.
# 내용(해시)과 파일명을 함께 비교한다. 이름만 바꿔 다시 올려도 출처 표기가 달라지므로 초기화한다.
doc_sig = tuple((d["id"], d["name"]) for d in docs)
if doc_sig != st.session_state.doc_sig:
    st.session_state.doc_sig = doc_sig
    st.session_state.messages = []
    st.session_state.analysis = None
    st.session_state.pending_question = None
st.session_state.docs = docs

# ---------------------------------------------------------------------------
# 업로드 요약
# ---------------------------------------------------------------------------
if not uploaded_files:
    # 빈 상태: 업로드 유도
    st.info("👆 분석할 PDF 파일을 업로드하면 시작할 수 있습니다.", icon="📂")
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip(
        (c1, c2, c3),
        [
            ("📝", "요약·분석", "전체 요약, 핵심 포인트, 문서 비교, 원하는 방식의 분석"),
            ("💬", "문서 Q&A", "문서 내용을 근거로 질문에 답하고 출처 페이지를 알려줍니다"),
            ("⬇️", "결과 저장", "분석 결과를 Markdown 파일로 내려받을 수 있습니다"),
        ],
    ):
        with col.container(border=True):
            st.markdown(f"### {icon}\n**{title}**")
            st.caption(desc)
else:
    total_tokens = sum(r["예상 토큰"] or 0 for r in rows)
    # 가로로 배치하고, 좁은 화면에서는 줄바꿈되도록 한다.
    with st.container(horizontal=True, wrap=True, gap="small"):
        for label, value in [
            ("분석 가능 파일", f"{len(docs)} / {len(rows)}"),
            ("총 페이지", f"{sum(d['pages'] for d in docs):,}"),
            ("총 글자 수", f"{sum(d['chars'] for d in docs):,}"),
            ("예상 토큰", f"{total_tokens:,}"),
        ]:
            st.metric(label, value, border=True, width=160)

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        placeholder="—",
        column_config={
            "페이지": st.column_config.NumberColumn(format="%d"),
            "글자 수": st.column_config.NumberColumn(format="localized"),
            "예상 토큰": st.column_config.NumberColumn(format="localized"),
            "비고": st.column_config.TextColumn(width="large"),
        },
    )

    if total_tokens > MAX_INPUT_TOKENS:
        st.caption(
            f"ℹ️ 문서 분량이 한 번에 처리할 수 있는 양({MAX_INPUT_TOKENS:,} 토큰)을 넘어 "
            "구간별로 요약한 뒤 분석합니다. 시간이 더 걸릴 수 있습니다."
        )

    if docs:
        with st.expander("📃 원문 미리보기"):
            names = [d["name"] for d in docs]
            selected = st.selectbox("파일 선택", names, label_visibility="collapsed") if len(docs) > 1 else names[0]
            doc = docs[names.index(selected)]
            st.text(doc["text"][:5000] + ("\n\n… (이하 생략)" if len(doc["text"]) > 5000 else ""))
    else:
        st.warning("분석할 수 있는 PDF가 없습니다. 텍스트가 포함된 PDF를 업로드해주세요.", icon="⚠️")

st.write("")
# key를 지정해 위쪽 요소 구성이 바뀌어도(파일 추가·경고 표시 등) 선택한 탭이 유지되게 한다.
tab_analysis, tab_qa = st.tabs(["📊 분석", "💬 Q&A 채팅"], key="main_tabs", on_change="rerun")

# ---------------------------------------------------------------------------
# 분석 탭
# ---------------------------------------------------------------------------
with tab_analysis:
    options = [t for t in ANALYSIS_TYPES if t != "문서 간 비교" or len(docs) >= 2]
    analysis_type = st.segmented_control(
        "분석 유형",
        options,
        default=options[0],
        required=True,
        format_func=lambda t: f"{ANALYSIS_ICONS[t]} {t}",
        disabled=not docs,
    )
    if docs and len(docs) < 2:
        st.caption("ℹ️ '문서 간 비교'는 PDF를 2개 이상 업로드하면 사용할 수 있습니다.")

    custom_prompt = ""
    if analysis_type == "사용자 지정":
        custom_prompt = st.text_area(
            "분석 요청 내용",
            placeholder="예: 이 문서에서 위험 요인과 대응 방안을 표로 정리해줘",
            disabled=not docs,
        )

    instruction = custom_prompt.strip() if analysis_type == "사용자 지정" else ANALYSIS_TYPES[analysis_type]
    run = st.button(
        "🚀 분석 시작",
        type="primary",
        # 사용자 지정 입력칸은 포커스를 잃어야 값이 반영되므로, 버튼을 비활성화하면 두 번 눌러야 한다.
        # 버튼은 항상 활성화하고 내용이 비었으면 안내한다.
        disabled=not docs,
        help=None if docs else "먼저 PDF 파일을 업로드하세요.",
    )
    if not docs:
        st.caption("PDF를 업로드하면 분석을 시작할 수 있습니다.")

    result_box = st.container()

    if run and not instruction:
        st.warning("분석 요청 내용을 입력해주세요.", icon="✏️")
    elif run:
        st.session_state.analysis = None
        with result_box:
            try:
                with st.status("문서 준비 중…", expanded=True) as status:
                    progress = st.empty()

                    def on_progress(done: int, total: int, level: int):
                        label = "긴 문서 구간별 요약 중" if level == 1 else f"요약본 재요약 중 ({level}단계)"
                        status.update(label=f"{label}… ({done}/{total})")
                        progress.progress(done / total if total else 1.0)

                    context, summarized = prepare_analysis_context(
                        api_key, model, docs, instruction, temperature, on_progress
                    )
                    progress.empty()
                    status.update(label="AI가 분석 결과를 작성하고 있습니다…")
                    messages = build_analysis_messages(context, instruction, language, summarized)
                    stream = stream_chat(api_key, model, messages, temperature)
                    first = next(stream, None)  # 첫 응답이 올 때까지 상태 표시 유지
                    status.update(label="분석 완료", state="complete", expanded=False)

                with st.container(border=True):
                    st.markdown(f"#### {ANALYSIS_ICONS[analysis_type]} {analysis_type} 결과")
                    text = stream_markdown(first, stream)
                if isinstance(text, str) and text.strip():
                    st.session_state.analysis = {"type": analysis_type, "text": text}
                else:
                    st.warning("모델이 빈 응답을 반환했습니다. 다시 시도해주세요.")
            except LLMError as e:
                status.update(label="분석 실패", state="error", expanded=False)
                st.error(f"분석 중 오류가 발생했습니다: {e}", icon="🚨")
    elif st.session_state.analysis:
        result = st.session_state.analysis
        with result_box, st.container(border=True):
            st.markdown(f"#### {ANALYSIS_ICONS.get(result['type'], '📊')} {result['type']} 결과")
            show_markdown(result["text"])

    if st.session_state.analysis:
        result = st.session_state.analysis
        _, right = st.columns([3, 1])
        right.download_button(
            "⬇️ 결과 다운로드 (.md)",
            data=f"# {result['type']}\n\n{result['text']}",
            file_name=f"pdf_analysis_{result['type'].replace(' ', '_')}.md",
            mime="text/markdown",
            on_click="ignore",
            width="stretch",
        )

# ---------------------------------------------------------------------------
# Q&A 탭
# ---------------------------------------------------------------------------
with tab_qa:
    # 대화 영역을 먼저 자리 잡고, 입력값을 읽은 뒤 내용을 채운다.
    # (입력값을 먼저 알아야 첫 질문 때 환영 메시지·예시 버튼이 남지 않는다.)
    chat_box = st.container()
    typed = st.chat_input(
        "문서에 대해 질문하세요" if docs else "먼저 PDF를 업로드하세요",
        disabled=not docs,
        max_chars=4000,
    )
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    with chat_box:
        if not docs:
            st.info("PDF를 업로드하면 문서 내용에 대해 질문할 수 있습니다.", icon="💬")
        elif not st.session_state.messages and not question:
            with st.chat_message("assistant", avatar=AI_AVATAR):
                st.markdown("업로드한 문서에 대해 무엇이든 물어보세요. 예시 질문을 눌러 시작할 수도 있어요.")
            cols = st.columns(len(EXAMPLE_QUESTIONS))
            for col, q in zip(cols, EXAMPLE_QUESTIONS):
                col.button(q, on_click=ask_example, args=(q,), width="stretch")

        for m in st.session_state.messages:
            avatar = USER_AVATAR if m["role"] == "user" else AI_AVATAR
            with st.chat_message(m["role"], avatar=avatar):
                show_markdown(m["content"])

    if question and docs:
        with chat_box:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(question)
            with st.chat_message("assistant", avatar=AI_AVATAR):
                try:
                    messages = build_qa_messages(docs, st.session_state.messages, question, language, model)
                    with st.spinner("답변을 생성하고 있습니다…"):
                        stream = stream_chat(api_key, model, messages, temperature)
                        first = next(stream, None)
                    answer = stream_markdown(first, stream)
                except LLMError as e:
                    st.error(str(e), icon="🚨")
                    answer = None
        # 실패한 질문은 이력에 남기지 않아 대화 맥락이 꼬이지 않게 한다.
        if isinstance(answer, str) and answer.strip():
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})
