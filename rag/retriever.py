import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient

from config import setup_settings, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


def _fetch_nodes_from_qdrant() -> list[TextNode]:
    """
    BM25 搜尋發生在本地記憶體，需要實際的文字內容作索引，單純只做 Qdrant 的連線是拿不到文字的
    所以要多一步從 Qdrant 把所有 nodes 撈回來
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    nodes = []
    offset = None

    while True:
        # points 就是 collection 內一個 node 的資料
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,         # Qdrant 撈資料要分批，每批的大小
            offset=offset,       # 下一批次要從哪裡開始撈
            with_payload=True,   
            with_vectors=False,  # 不用向量資料
        ) 

        for point in points:
            node_content = point.payload.get("_node_content")

            if node_content:
                nodes.append(TextNode.from_json(node_content))

        if next_offset is None:
            break
        offset = next_offset

    return nodes


def build_retriever(index: VectorStoreIndex, top_k: int = 5) -> QueryFusionRetriever:
    """
    - 建立 Hybrid Search Retriever，這裡結合了 VectoreSearch 和 BM25 兩種方式做檢索，技術文件通常充滿轉有名詞，故要同時兼顧語意與關鍵字
    - 為了避免不同 retriever 分數尺度不同，這裡使用 RRF 演算法，根據各自 retrievers 排名計算合併分數
      RRF 公式： 某個 chunk 的 score = 1/(k + rank1) + 1/(k + rank2) + ... + 1/(k + rankN)，N = retrievers 數量
    """

    # 做語意搜尋 (consine similarity)，計算餘弦相似度
    # 流程: 使用者問題 -> embedding 成向量 -> 送到 Qdrant -> Qdrant 在雲端計算相似度 -> 回傳 Top-K
    vector_retriever = VectorIndexRetriever(
        index=index,  # 需要的是向量化後可供查詢的 inedex
        similarity_top_k=top_k,
    )
    # 取得每個 chunk 的原始文本
    nodes = _fetch_nodes_from_qdrant()
    # TF-IDF 的進化版，根據原始文件 (非向量) 詞彙出現頻率和文件長度計分
    # 流程: 使用者問題 -> 分詞 -> 對本地所有 nodes 計算 TF-IDF 分數 -> 本地暴力計算 -> 回傳 Top-K
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,   # 需要的是原始資料的 node
        similarity_top_k=top_k,
    )

    # QueryFusionRetriever 能把多個 retriever 的結果合併
    retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,  # 代表只用原始問題去查詢
        mode="reciprocal_rerank",  # 使用 RFF 演算法
        use_async=False
    )

    return retriever

if __name__ == "__main__":
    from rag.indexer import load_index

    index = load_index()
    retriever = build_retriever(index, top_k=5)

    query = "What is LlamaIndex used for?"
    nodes = retriever.retrieve(query)

    print(f"\n查詢: {query}")
    print(f"\n找到 {len(nodes)} 個相關 chunks")

    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_name", "未知來源")
        score = round(node.score, 4) if node.score else 0.0
        print(f"--- [{i}] 來源: {source}  分數: {score} ---")
        print(node.text[:200])
        print()
