"""
LangChain Brain 聊天介面
執行方式：streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).parent))

# --- 頁面基本設定（必須是第一個 Streamlit 指令）---
st.set_page_config(
    page_title="LangChain Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",  # 預設展開側邊欄
)

# --- 自訂樣式 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* 整體背景 */
.stApp {
    background-color: #FCFCFC;
    font-family: 'Inter', sans-serif;
}

/* 側邊欄 */
[data-testid="stSidebar"] {
    background-color: #F0F0F0;
    border-right: 1px solid #d0d7de;
}

/* 輸入框 */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    border: 1px solid #d0d7de !important;
    color: #24292f !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
}

/* 使用者訊息泡泡 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    padding: 12px 16px;
}

/* AI 訊息泡泡 */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #ffffff;
    border-left: 3px solid #2da44e;
    padding: 12px 16px;
}

/* 標題樣式 */
h1 {
    font-family: 'JetBrains Mono', monospace !important;
    color: #24292f !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.5px;
}

h3 {
    color: #57606a !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px;
}

/* 來源引用區塊 */
.source-box {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-left: 3px solid #0969da;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 6px;
    font-size: 0.8rem;
    color: #57606a;
    font-family: 'JetBrains Mono', monospace;
}

.source-box strong {
    color: #0969da;
}

/* 狀態 badge */
.status-badge {
    display: inline-block;
    background-color: #2da44e;
    color: #ffffff;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.status-badge-warning {
    background-color: #bf8700;
}

/* 側邊欄文字 */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #3C3C3C !important;
    font-size: 0.95rem !important;
}

[data-testid="stSidebar"] h2 {
    color: #0969da !important;
}

/* 分隔線 */
hr {
    border-color: #d0d7de !important;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #0969da !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* 按鈕 */
.stButton button {
    background-color: #66B3FF !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# --- 載入 RAG Pipeline ---

# 第一次載入頁面 → 執行函式，結果存進快取，之後每次重新整理 → 直接用快取，不重新執行
# 只有呼叫 _build_engine.clear() → 清除快取，下次重新執行
@st.cache_resource(show_spinner=False)   #  可以快取 function 的返回結果，讓頁面重新渲染時直接拿
def _build_engine(answer_llm):
    """連接 Qdrant collection 並建立 QueryEngine，有新文件 ingest 後需清除此快取"""
    from config import setup_settings
    from rag.indexer import collection_exists, load_index
    from rag.pipeline import build_query_engine
    setup_settings(answer_llm)   # 前端使用者可自由選回答模型

    # 知識庫尚未建立的階段，要先按同步按鈕建立
    # 注意!!!!其實這個判斷警示不應該顯示在前端，因為建立知識庫是我們開發者事先就要全部建立好，而不是交由使用者按鈕 (但還是保留避免例外)
    if not collection_exists():
        raise RuntimeError("知識庫尚未建立，請按「🔄 同步新文件」按鈕進行初始化")
    
    # 已經有知識庫 collection，載入並建立成 index
    index = load_index(answer_llm)
    return build_query_engine(index, top_k=5)

# --- 初始化 Session State ---
# st.session_state 是全域狀態容器，用在每次重新渲染間保留資料狀態
if "messages" not in st.session_state:
    st.session_state.messages = []   # 當前對話的所有訊息

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True   # 是否顯示來源引用


# ---側邊欄，進入頁面後會有一系列工作執行，會在側邊欄顯示 ---
with st.sidebar:
    # 只在第一次初始化 RAG 回答模型，之後 render 不覆寫
    if "llm_option" not in st.session_state:
        st.session_state.llm_option = "OpenAI/gpt-4o-mini"

    # 載入狀態
    st.markdown("## SYSTEM")
    is_ready = False  # 預設系統未就緒
    error_msg = None  # 錯誤訊息
    query_engine = None  # 預設查詢引擎狀態

    try:
        # 1. 先連接 Qdrant collection 建立 QueryEngine (不論 collection 是否存在)
        with st.spinner("連接知識庫中..."):  # st.spinner 顯示旋轉動畫模擬載入中
            query_engine = _build_engine(st.session_state.llm_option)
        # 如果成功連接，代表系統就緒
        is_ready = True
    except Exception as e:
        error_msg = str(e)

    if is_ready:
        st.markdown('<span class="status-badge">● ONLINE</span>', unsafe_allow_html=True)
        # 顯示切換模型的提示
        if "model_switched" in st.session_state:
            # 每次顯示後就要 pop() 清空狀態，這樣下一次切換才可以重新將 model_switched 加入 
            st.success(f"已切換至 {st.session_state.pop('model_switched')}")
        # 顯示上一次同步結果（按鈕觸發後由 session_state 傳入）
        if "sync_result" in st.session_state:
            result = st.session_state.pop("sync_result")
            # 有無新文件進入是不同的 count
            if result["count"] > 0:
                st.info(f"已同步 {result['count']} 份新文件")
            else:
                st.caption("知識庫已是最新，無新文件")
    # 如果連接失敗，顯示錯誤訊息
    else:
        st.markdown('<span class="status-badge status-badge-warning">● OFFLINE</span>', unsafe_allow_html=True)
        st.warning(error_msg)

    st.markdown("---")

    # 可自由切換回答模型 (非向量化與檢索)
    _llm_options = ["OpenAI/gpt-4o-mini", "Anthropic/claude-sonnet-4-6"]
    llm_option = st.selectbox(
        "回答模型",
        options=_llm_options,
        index=_llm_options.index(st.session_state.llm_option),
    )

    # 跟上次選擇比較，只有改變才觸發
    if llm_option != st.session_state.llm_option:
        st.session_state.llm_option = llm_option
        st.session_state["model_switched"] = llm_option
        st.rerun()  # 重跑頁面就會重新執行 _build_engine()，看是拿快取還是重建

    # 設定
    st.markdown("## SETTINGS")
    st.session_state.show_sources = st.toggle(   # 每次按 toggle 都是重新渲染頁面一次
        "顯示來源引用",
        value=st.session_state.show_sources,
    )

    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="每次查詢取回幾個相關 chunk",
    )

    st.markdown("---")

    # --- 知識庫同步 (點同步按鈕就執行) ---
    st.markdown("## SYNC")  
    if st.button("🔄 同步新文件", use_container_width=True):
        from config import setup_settings
        from ingestion.sync_docs import detect_new_docs, sync_new_docs
        setup_settings(st.session_state.llm_option)
        try:
            with st.spinner("比對是否存在新文件中..."):
                # 先判斷是否存在 Drive 中有新文件
                new_files = detect_new_docs()

            if not new_files:
                st.session_state["sync_result"] = {"count": 0}
                st.rerun()    # 重新渲染頁面
            else:
                with st.spinner(f"偵測到新文件，重建知識庫中..."):
                    # 有新文件就執行 Ingestion 步驟，重建知識庫
                    count = sync_new_docs(new_files, st.session_state.llm_option)
                _build_engine.clear()    # 重建知識庫後要清除原先的快取，否則還是舊的 query_engine
                st.session_state["sync_result"] = {"count": count}
                st.rerun()
        except Exception as e:
            st.error(f"同步失敗：{str(e)}")

    st.markdown("---")

    # 清除對話
    if st.button("🗑 清除對話記錄", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # 範例問題
    st.markdown("## EXAMPLES")
    example_questions = [
        "What is LangChain?",
        "How does memory work in LangChain?",
        "What is LangGraph used for?",
        "How to use ConversationBufferMemory?",
        "What are LangChain agents?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_question = q
            st.rerun()


# --- 主畫面 ---
# 這裡要注意一點，所有對話的內容都是 st.markdown() 去渲染，因此原生 LLM 輸出的 md 回覆都會正常顯示
st.markdown("# 🧠 LangChain Brain")
st.markdown(
    "LangChain / LangGraph / LangSmith 知識庫問答系統",
)
st.markdown("---")

# 顯示歷史對話 (遞迴顯示使用者與 AI 的訊息)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])   # 顯示訊息內容

        # 顯示來源（只有 assistant 訊息才有）
        if (
            message["role"] == "assistant"
            and st.session_state.show_sources  # 有點選顯示來源
            and "sources" in message
            and message["sources"]
        ):
            with st.expander(f"📎 來源引用（{len(message['sources'])} 個）"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>[{i}]</strong> {source["file"]} '
                        f'<span style="color:#3fb950">RFF score: {source["score"]:.3f}</span><br>'
                        f'<span style="color:#6e7681">{source["preview"]}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# --- 處理範例問題點擊 ---
pending = st.session_state.pop("pending_question", None)   # 取出範例問題，且要清除，因為是一次性的


# --- 問答邏輯 ---
def process_question(question: str):
    """處理使用者問題，呼叫 RAG pipeline 並顯示結果"""

    # 顯示使用者訊息
    with st.chat_message("user"):
        st.markdown(question)   # 這邊立刻顯示，不等 RAG

    # 將所有訊息存進 session state，方便顯示 (也讓前面函式可以取出使用)
    st.session_state.messages.append({"role": "user", "content": question})

    # 呼叫 RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("搜尋知識庫中..."):
            try:
                response = query_engine.query(question)
                answer = str(response)

                # 萃取來源資訊
                sources = []
                if hasattr(response, "source_nodes"):
                    for node in response.source_nodes:
                        sources.append({
                            "file": node.metadata.get("file_name", "未知來源"),
                            "score": node.score or 0.0,
                            "preview": node.text[:120].replace("\n", " ") + "...",
                        })

                st.markdown(answer)

                # 顯示來源
                if st.session_state.show_sources and sources:
                    with st.expander(f"📎 來源引用（{len(sources)} 個）"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-box">'
                                f'<strong>[{i}]</strong> {source["file"]} '
                                f'<span style="color:#3fb950">score: {source["score"]:.3f}</span><br>'
                                f'<span style="color:#6e7681">{source["preview"]}</span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # 儲存到 session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_text = f"查詢失敗：{str(e)}"
                st.error(error_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                })


# 1. 處理範例問題
if pending and is_ready:
    process_question(pending)  # 當成一般的問題丟進去呼叫 RAG

# 2. 處理使用者輸入
question = st.chat_input("問題...")  
if question:                          # 再判斷
    process_question(question)


# --- 空白狀態提示 ---
if not st.session_state.messages and is_ready:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #57606a;">
        <div style="font-size: 3rem; margin-bottom: 16px;">🧠</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #24292f;">
            LangChain 知識庫已就緒
        </div>
        <div style="font-size: 0.85rem; margin-top: 8px;">
            在下方輸入問題，或點選左側範例問題開始
        </div>
    </div>
    """, unsafe_allow_html=True)