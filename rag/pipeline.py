import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from rag.retriever import build_retriever
from llama_index.core import Settings


"""
把 retriever 和 LLM 串起來，提供一個「問問題 → 得到答案 + 來源」的介面
"""

# # 這個只適用於單輪對話 (目前系統版本停用)
# def build_query_engine(index: VectorStoreIndex, top_k: int = 10) -> RetrieverQueryEngine:
#     """
#     把 retriever.py 內建好的 retriever 包進 RetrieverQueryEngine，再加上 response_synthesizer (把所有 chunks 合起來)
#     - top_k: 從 Hybrid Search 撈出相關 chunks 數量、後續 rerank 的數量
#     """
    
#     # 建立可以執行 chunk 搜尋的 retriever
#     retriever = build_retriever(index, top_k)

#     # 建立用 Cross-Encoder 進行 rerank 的 reranker
#     # Cross-Encoder 會將 query 與 chunks 同時送進 transformer，直接輸出相關性分數，精度更高
#     reranker = SentenceTransformerRerank(
#         model="BAAI/bge-reranker-v2-m3",  # 適合處理中文
#         top_n=top_k,
#     )

#     # LlamaIndex 提供的回答生成器，負責接收 query 與檢索到的 chunls，回答模型會從全域 Settings 取得
#     # "compact"：把所有 chunk 塞進一個 prompt，一次生成答案（省錢）
#     # "refine"：逐一處理每個 chunk，逐步修正答案（品質較高但貴）
#     # "tree_summarize"：先對每組 chunk 摘要，再合併（適合超長文件）
#     response_synthesizer = get_response_synthesizer(response_mode="compact")
    
#     # 建成一個具有檢索功能，以及能回答問題的 engine
#     return RetrieverQueryEngine(
#         retriever=retriever,
#         response_synthesizer=response_synthesizer,
#         node_postprocessors=[reranker],  # ← Retriever → Reranker → LLM
#     )

def build_multi_turn_chat_engine(index: VectorStoreIndex, top_k: int = 10):
    """
    建構具有多輪對話、記憶功能的問答引擎
    - CondensePlusContextChatEngine 會將歷史對話加上當前 query，做為 user prompt 去檢索，並且將結果與歷史對話提供給回答模型
    """
    
    # 建立可以執行 chunk 搜尋的 retriever
    retriever = build_retriever(index, top_k)

    # 建立用 Cross-Encoder 進行 rerank 的 reranker
    # Cross-Encoder 會將 query 與 chunks 同時送進 transformer，直接輸出相關性分數，精度更高
    reranker = CohereRerank(
        api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-multilingual-v3.0",  # 支援中文
        top_n=10,
    )

    SYSTEM_PROMPT = """你是使用者的個人知識助理，具備專業性、親切性，專門依據他存放於 HackMD 的技術筆記進行回答。
        回答原則：
        - 所有回答必須以筆記中的內容為依據，不要憑空捏造或依賴訓練資料的既有記憶
        - 若筆記內容足以回答，給出清楚、詳細、有條理的回答
        - 若筆記內容僅部分相關，整合現有片段給出最佳答案，並說明資訊有限
        - 解釋技術概念時，適時使用生活化比喻幫助理解
        - 語氣像一位熟悉使用者筆記內容的私人技術顧問，親切且專業

        回答格式（每次回答都必須遵守）：
        1. 【結論】先用 1-2 句話直接給出核心答案
        2. 【說明】列出具體步驟、原理、或補充細節
        3. 【注意事項】補充使用限制、常見踩坑點、或延伸建議

        如果知識庫中完全找不到相關資料，請回覆：
        「嗯⋯這題難倒我了，您的知識庫裡暫時找不到相關筆記。建議您可以：
        1. 換個關鍵字重新提問
        2. 聯絡開發人員將相關資料補充到 HackMD 後再同步知識庫」
    """

    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=Settings.llm,    # 要傳入模型參數，因為使用者可在前端自由切換
        system_prompt=SYSTEM_PROMPT,
        node_postprocessors=[reranker],  # ← Retriever → Reranker → LLM
        verbose=False,
    )

def query_with_sources(engine: RetrieverQueryEngine, question: str) -> str:

    # 執行查詢產生回答的資訊
    response = engine.query(question)

    print(f"\n答案: \n{response.response}")

    print("\n來源:")
    # 每個 chunk 的來源 
    seen = set()

    for node in response.source_nodes:
        # 這個 chunk (node) 來自哪個檔案，以及他在 Drive 的路徑
        source = node.metadata.get("file_name", node.metadata.get("file_path", "未知來源"))
        score = round(node.score, 4) if node.score else 0.0

        if source not in seen:
            print(f" - {source} (分數: {score})")
            # 有可能很多 chunk 來自同一份文件，因此只要印一次就好
            seen.add(source)

    return response.response

if __name__ == "__main__":
    from rag.indexer import collection_exists, build_index, load_index

    index = load_index()
    engine = build_query_engine(index)
    question = "What is LlamaIndex used for?"
    response = query_with_sources(engine, question)