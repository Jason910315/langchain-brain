import os
from dotenv import load_dotenv
from llama_index.core import Settings

load_dotenv()

"""
將 RAG 內所需要的設定都在此程式管理
"""
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai") # 嵌入的模型

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 切換 LLM 時 embedding 維度不同，不能共用同一個 collection，要重建
COLLECTION_NAME = f"langchain_docs_{EMBEDDING_PROVIDER}"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

def get_llm(llm):
    """
    可以替換不同的回答模型 (並不做檢索與 Embedding，他負責接收檢索回來的資料並回答)
    """
    if llm == "OpenAI/gpt-4o-mini":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif llm == "Anthropic/claude-sonnet-4-6":
        from llama_index.llms.anthropic import Anthropic
        return Anthropic(
            model="claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    else:
        raise ValueError(f"不支援的回答模型: {llm}")

def get_embedding():
    """
    取得 embedding 與 retrieval 使用的模型
    """
    if EMBEDDING_PROVIDER == "text-embedding-3-small":
          from llama_index.embeddings.openai import OpenAIEmbedding
          return OpenAIEmbedding(
              model="text-embedding-3-small",
              api_key=os.environ["OPENAI_API_KEY"],
          )
    elif EMBEDDING_PROVIDER == "qwen":
        # [學習] OpenAILikeEmbedding 專為 OpenAI-compatible 第三方 API 設計，
        # 不做 model 名稱 enum 驗證，適合 OpenRouter 等代理服務
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding
        return OpenAILikeEmbedding(
            model_name="qwen/qwen3-embedding-8b",
            api_key=os.environ["OPENROUTER_API_KEY"],
            api_base="https://openrouter.ai/api/v1",
            embed_batch_size=10,
        )
    else:
          raise ValueError(f"不支援的 EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

# 這邊回答模型用傳入的，讓前端使用者自由選取模型後，app.py 傳入
def setup_settings(answer_llm):
    # LlamaIndex 的 Settings 是全域設定物件
    Settings.llm = get_llm(answer_llm)
    Settings.embed_model = get_embedding()


if __name__ == "__main__":
    setup_settings()
    print(f"回答模型      : {llm}")
    print(f"EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
    print(f"LLM               : {Settings.llm.__class__.__name__}")
    print(f"Embedding         : {Settings.embed_model.__class__.__name__}")
    print(f"Collection        : {COLLECTION_NAME}")