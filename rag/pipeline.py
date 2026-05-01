import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from rag.retriever import build_retriever


"""
把 retriever 和 LLM 串起來，提供一個「問問題 → 得到答案 + 來源」的介面
"""

def build_query_engine(index: VectorStoreIndex, top_k: int = 5) -> RetrieverQueryEngine:
    """
    把 retriever.py 內建好的 retriever 包進 RetrieverQueryEngine，再加上 response_synthesizer (把所有 chunks 合起來)
    """
    
    # 建立可以執行 chunk 搜尋的 retriever
    retriever = build_retriever(index, top_k)

    # LlamaIndex 提供的回答生成器，負責接收 query 與檢索到的 chunls，回答模型會從全域 Settings 取得
    # "compact"：把所有 chunk 塞進一個 prompt，一次生成答案（省錢）
    # "refine"：逐一處理每個 chunk，逐步修正答案（品質較高但貴）
    # "tree_summarize"：先對每組 chunk 摘要，再合併（適合超長文件）
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    
    # 建成一個具有檢索功能，以及能回答問題的 engine
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

def build_multi_turn_chat_engine(index: VectorStoreIndex, top_k: int = 5):
    """
    建構具有多輪對話、記憶功能的問答引擎
    - CondensePlusContextChatEngine 會將歷史對話加上當前 query，做為 user prompt 去檢索，並且將結果與歷史對話提供給回答模型
    """
    
    # 建立可以執行 chunk 搜尋的 retriever
    retriever = build_retriever(index, top_k)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
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