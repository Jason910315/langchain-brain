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
COLLECTION_NAME = f"personal_notes_{EMBEDDING_PROVIDER}"

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
    elif llm == "ft:gpt-4o-mini-2024-07-18:personal::DeCCm77q":
        from llama_index.llms.openai import OpenAI
        # OpenAI 微調後的模型呼叫方式跟普通 OpenAI 模型一樣
        # [重要] 微調模型需要明確設定 system_prompt，否則 LlamaIndex 預設 prompt 會蓋過微調行為
        # 讓微調模型知道自己是 RAG 問答助理，context 會由系統注入，不要靠訓練記憶回答
        return OpenAI(
            model="ft:gpt-4o-mini-2024-07-18:personal::DeCCm77q",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt=(
               """你是一位專精於 LangChain / LangGraph / LangSmith 的技術助理。
               請用以下格式回答：
               1. 先給出結論（1-2 句話）
               2. 再列出具體步驟或說明
               3. 最後補充原理或注意事項

               回答要專業、清楚，適時用生活化比喻幫助理解。
               如果文件內容不足以回答這個問題，請回覆:「嗯⋯這題難倒我了，知識庫裡找不到相關資料。你可以嘗試：換個問法重新提問，或直接查閱官方文件，或許答案就在那裡等你！
               也可以聯繫管理員補充相關文件。」
            """
            ),
        )
        
    else:
        raise ValueError(f"不支援的回答模型: {llm}")

def get_embedding():
    """
    Qdrant 的 collection 是獨立的，你讀取他就代表他檢索使用的會跟 embedding 時一樣的模型，因此你不用額外設定。
    這個函式取得 embedding 與 retrieval 使用的模型，在 .env 切換後所有程式都會從 Settings.embed_model 取得模型。
    """
    from llama_index.embeddings.openai import OpenAIEmbedding
    return OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )


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