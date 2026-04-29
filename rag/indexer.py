import sys
from pathlib import Path

# 把專案根目錄加進 Python 的模組搜尋路徑
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, setup_settings
)

"""
這個程式會將傳進來的 Documents 物件自動地做完整的向量化 pipeline，讓他能存到知識庫
"""

def _get_qdrant_client() -> QdrantClient:

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,)

def collection_exists() -> bool:
    """
    檢查 Qdrant collection 是否已經存在而且有資料，給 main.py 判斷要 build 還是 load
    避免每次啟動都要重建 index
    """
    client = _get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return False
    
    return client.count(collection_name=COLLECTION_NAME).count > 0

# --- 分為兩個分支 : build & load ---
# 返回的都是 Index 物件，他是一整個包含所有 chunk 且可供查詢的物件
def build_index(documents: list[Document], answer_llm: str) -> VectorStoreIndex:
    """
    起始沒有 collection，需要新建立 index 時使用: 切 chunk -> embedding -> 存入 Qdrant
    Args: 
        documents: 接收一個 Document 清單，包含所有要向量化的資料 (一份或很多份文件)
    """
    setup_settings(answer_llm) # 先設置回答以及嵌入模型

    client = _get_qdrant_client()
    # 建立儲存向量資料的連線
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    # 建立完整 pipeline，會依序執行切 chunk -> embedding -> 存入 Qdrant
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            Settings.embed_model,  # 向量化的模型
        ],
        vector_store=vector_store,
    )

    # node 就是大家說的 chunk，執行完 run 就會向量化並開一個新的 vector_store (collection) 儲存
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"\nIndex 建立完成，共 {len(nodes)} 個 chunks 存入 {COLLECTION_NAME}")
    
    # 返回的是一個 index 物件，可供查詢 
    return VectorStoreIndex.from_vector_store(vector_store)

def load_index(answer_llm: str) -> VectorStoreIndex:
    """
    已經有 collection 了，不需重建，直接載入
    """
    setup_settings(answer_llm) 
    client = _get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print(f"載入既有的 Index [{COLLECTION_NAME}]")
    return VectorStoreIndex.from_vector_store(vector_store)

def insert_documents(index: VectorStoreIndex, documents: list[Document]) -> None:
    """
    未來想新增文件或 github issue 資料時使用，直接增量插入，不需將整個 collection 重建
    """
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            Settings.embed_model,
        ],
        vector_store=index.storage_context.vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"增量插入完成，共新增 {len(nodes)} 個 chunk")
    

if __name__ == "__main__":
    client = _get_qdrant_client()

    doc1 = Document(
        text="LlamaIndex is a data framework for LLM applications. " * 40,
        metadata={"file_name": "doc1.md", "source_folder": "test"},
        doc_id="local-test-001",
    )

    doc2 = Document(
        text="Qdrant is a vector database optimized for similarity search. " * 40,
        metadata={"file_name": "doc2.md", "source_folder": "test"},
        doc_id="local-test-002",
    )

    # --- 測試 1: build_index ---
    print("=== 測試 1: build_index ===")
    index = build_index([doc1])
    count_after_build = client.count(collection_name=COLLECTION_NAME).count
    print(f"建立後 chunk 總數: {count_after_build}")
    assert count_after_build > 0, "build_index 失敗：Qdrant 沒有資料"
    print("測試 1 通過\n")

    # --- 測試 2: insert_documents ---
    print("=== 測試 2: insert_documents ===")
    count_before = client.count(collection_name=COLLECTION_NAME).count
    insert_documents(index, [doc2])
    count_after = client.count(collection_name=COLLECTION_NAME).count
    print(f"插入前 chunk 數: {count_before}，插入後: {count_after}")
    assert count_after > count_before, "insert_documents 失敗：chunk 數沒有增加"
    print("測試 2 通過")