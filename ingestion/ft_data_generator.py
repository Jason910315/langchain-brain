import sys, os, json, random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from openai import OpenAI
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """你是一位專精於 LangChain / LangGraph / LangSmith 的技術助理。
回答時以專業工程師的角度，解釋概念或原理時要祥細清楚，並適時搭配生活化的比喻幫助理解。

回答格式:
1. 先給出結論
2. 再列出具體步驟或說明
3. 最後補充原理或注意事項

你只回答與 LangChain / LangGraph / LangSmith 相關的問題。
如果文件內容不足以回答這個問題，請回覆:「嗯⋯這題難倒我了，知識庫裡找不到相關資料。你可以嘗試：換個問法重新提問，或直接查閱官方文件，或許答案就在那裡等你！
也可以聯繫管理員補充相關文件。」
"""

def build_user_message(question: str, chunks: list[str]) -> str:
    chunk_text = "\n\n".join(chunks)
    return f"以下是相關文片段:\n\n{chunk_text}\n\n問題: {question}"

# 超出範圍的問題清單（只需要問題，答案固定是拒答）
OUT_OF_SCOPE_QUESTIONS = [
    "Python 的 list comprehension 怎麼用？",
    "FastAPI 怎麼部署到 AWS？",
    "React 的 useEffect 什麼時候會觸發？",
    "MySQL 和 PostgreSQL 哪個比較好？",
    "今天天氣怎麼樣？",
    "Docker Compose 怎麼設定環境變數？",
    "GPT-4 和 Claude 哪個比較聰明？",
    "怎麼學好英文？",
    "Kubernetes 的 Pod 和 Deployment 差在哪？",
    "幫我寫一段爬蟲程式",
    "怎麼用 Pandas 讀取 CSV 檔案？",
    "Git rebase 和 merge 差在哪？",
    "TypeScript 的 interface 和 type 差在哪？",
    "Redis 的 TTL 怎麼設定？",
    "Linux 怎麼查看 CPU 使用率？",
    "幫我推薦一個好的咖啡廳",
    "今天股市怎麼樣？",
    "怎麼減肥？",
    "Vue 和 React 哪個比較好學？",
    "Nginx 怎麼設定反向代理？",
    "幫我寫一封英文求職信",
    "SQL 的 JOIN 有幾種？",
    "怎麼用 curl 發送 POST request？",
    "幫我解釋牛頓第二定律",
    "MongoDB 和 MySQL 差在哪？",
    "怎麼在 Mac 上安裝 Homebrew？",
    "幫我寫一個排序演算法",
    "Terraform 怎麼管理 AWS 資源？",
    "CI/CD pipeline 怎麼設計？",
    "怎麼用 Selenium 做自動化測試？",
]

# 統一拒答的格式
REFUSAL_ANSWER = "嗯⋯這題難倒我了，知識庫裡找不到相關資料。你可以嘗試：換個問法重新提問，或直接查閱官方文件，或許答案就在那裡等你！也可以聯繫管理員補充相關文件。\n\n"

# 從 Qdrant 撈出所有 chunks
def fetch_chunks_from_qdrant() -> list[str]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    texts = []
    offset = None

    print("從 Qdrant 撈取 chunks 原文...")

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            node_content = point.payload.get("_node_content")
            if node_content:
                node = TextNode.from_json(node_content)
                # 過濾掉長度太短的 chunk，沒有訓練價值
                if len(node.text.strip()) > 100:
                    texts.append(node.text.strip())
        # 已經批次撈完所有 chunk 了
        if next_offset is None:
            break
    
        offset = next_offset
    
    print(f"共撈取 {len(texts)} 個有效 chunks 原文")
    return texts

def generate_question_for_chunks(client: OpenAI, chunk_text: str) -> list[str]:
    """
    針對單一 chunk 的內容，用 gpt-4o-mini 自動生成兩個問題，做為 fine-tuning 的訓練資料
    """
    prompt = f"""以下是一段 LangChain / LangGraph / LangSmith 的技術文件內容:
    {chunk_text}

    請根據這段內容，生成 2 個使用者可能會問的問題。
    要求:
    - 問題要自然，像真實使用者在提問
    - 一個問技術概念，一個問實際用法或步驟
    - 直接輸出問題，每行一個，不要編號或其他格式

    只輸出問題，不要有其他說明。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()

        # 將每個問題列出來，並只取前兩個 (防禦性寫法，這裡強制隔開與取兩個提問)
        questions = [q.strip() for q in raw.split("\n") if q.strip()]
        print(f"生成問題: {questions}")
        return questions[:2]

    except Exception as e:
        print(f"生成問題失敗: {e}")
        return []

def generate_answer_for_qa(client: OpenAI, question: str, chunk_text: str) -> str:
    """
    針對問題 + chunk 內容，生成符合你定義格式的標準答案
    這個答案就是 Fine-tuning 的「標準答案」，模型會學習「看到這類問題，要用這種格式回答」
    """

    prompt = f"""你是一位專精於 LangChain / LangGraph / LangSmith 的技術助理。
        
        請跟據以下技術文件內容回答問題：

        文件內容：
        {chunk_text}

        問題：{question}

        請用以下格式回答：
        1. 先給出結論（1-2 句話）
        2. 再列出具體步驟或說明
        3. 最後補充原理或注意事項

        回答要專業、清楚，適時用生活化比喻幫助理解。
        如果文件內容不足以回答問題，請說「根據現有文件資料不足以完整回答此問題」。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # 低一點讓答案格式更穩定
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成答案失敗：{e}")
        return ""

def build_training_example(question: str, answer: str, chunks: list[str]) -> dict:
    """
    把一組 QA 包裝成 OpenAI Fine-tuning 的訓練資料格式，定義模型的「輸入-輸出」範本格式
    訓練時的訊息格是要跟推理時完全一致，也就是不論是正確答案或拒答，都要用這個格式
    """
    # 把問題 + chunk 原文包裝成 user message，讓模型知道這個問題的背景
    user_message = build_user_message(question, chunks)

    # OpenAI Fine-tuning 的資料格式固定是這樣
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
    }

def main():
    # 包成 jsonl 格式，每一行都是獨立的 QA
    output_path = Path("data/ft_train.jsonl")
    output_path.parent.mkdir(exist_ok=True)  # 如果資料夾不存在，則建立

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_examples = []

    all_chunks = fetch_chunks_from_qdrant()
    if not all_chunks:
        print("錯誤: Qdrant 內沒有資料，請先執行同步")

    # 隨機抽樣 80 個 chunks 來生成訓練資料，這樣就會有 160 筆訓練資料
    sampled_chunks = random.sample(all_chunks, min(80, len(all_chunks)))

    print(f"\n開始生成訓練資料 (總共 {len(sampled_chunks)} 個 chunks)...")

    for i, chunk in enumerate(sampled_chunks, start=1):
        print(f"[{i}/{len(sampled_chunks)}] 生成問題中....")

        # 根據這個 chunk 的原文，生成 2 個問題
        questions = generate_question_for_chunks(openai_client, chunk)

        for question in questions:
            if not question:
                continue
            # 根據問題 + chunk 原文，生成符合你定義格式的標準答案
            answer = generate_answer_for_qa(openai_client, question, chunk)

            if answer:
                # 將 QA 包裝成 Fine-tuning 的訓練資料格式
                all_examples.append(build_training_example(question, answer, [chunk]))

    print("有答案資料生成: ", len(all_examples) ,"筆")

    print(f"開始生成拒答訓練資料 {len(OUT_OF_SCOPE_QUESTIONS)} 筆...")

    for question in OUT_OF_SCOPE_QUESTIONS:
        # 拒答資料產生的情況是 chunks 是從知識庫隨機撈的不相關內容
        # 模擬真實，RAG 撈回來的 chunks 跟問題對不上，所以模型要學會「看到不相關的 chunks + 超出範圍的問題 → 要拒答」
        random_chunks = random.sample(all_chunks, min(5, len(all_chunks)))
        answer = REFUSAL_ANSWER
        all_examples.append(build_training_example(question, answer, random_chunks))


    # # 打亂所有問題的順序，避免模型學到，前幾筆都是有答案，後幾筆都是拒答的順序偏差
    # random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            # 每一筆訓練資料都用換行隔開，並轉成 json 格式
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    total = len(all_examples)
    has_answer = total - len(OUT_OF_SCOPE_QUESTIONS)
    print(f"\n✅ 完成！共 {total} 筆訓練資料")
    print(f"   有答案：{has_answer} 筆")
    print(f"   拒答：{len(OUT_OF_SCOPE_QUESTIONS)} 筆")
    print(f"   輸出位置：{output_path}")

if __name__ == "__main__":
    main()