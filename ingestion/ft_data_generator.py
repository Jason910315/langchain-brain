import sys, os, json, random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from openai import OpenAI
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """您是使用者的個人知識助理，具備專業性、親切性，專門依據他存放於 HackMD 的技術筆記進行回答。
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

def build_user_message(question: str, chunks: list[str]) -> str:
    chunk_text = "\n\n".join(chunks)
    return f"以下是相關文片段:\n\n{chunk_text}\n\n問題: {question}"

# 超出範圍的問題清單（只需要問題，答案固定是拒答）
OUT_OF_SCOPE_QUESTIONS = [
    # 生活與娛樂類
    "今天晚餐吃什麼好？",
    "幫我推薦一部好看的日劇",
    "台北哪裡有好吃的火鍋？",
    "怎麼減肥最有效？",
    "今天股市漲跌如何？",
    "幫我寫一首情歌歌詞",
    "明天天氣怎麼樣？",
    "怎麼規劃日本京都旅遊行程？",
    "買房還是租房比較划算？",
    "怎麼挑選一台好的咖啡機？",

    # 筆記完全未涵蓋的技術領域
    "Unity 怎麼做 3D 物理碰撞？",
    "Swift 的 async/await 怎麼用？",
    "Flutter 怎麼做狀態管理？",
    "怎麼用 Blender 做 3D 建模？",
    "Solidity 智慧合約怎麼部署到以太坊？",
    "Unreal Engine 的 Blueprint 怎麼用？",
    "Arduino 怎麼控制伺服馬達？",
    "Terraform 怎麼管理 AWS 資源？",
    "Kubernetes 的 Pod 和 Deployment 差在哪？",
    "Ansible 怎麼寫 playbook？",

    # 醫療、法律、財務類
    "我最近頭很痛，可能是什麼病？",
    "離婚要怎麼分財產？",
    "如何申報綜合所得稅？",
    "比特幣現在值得買嗎？",
    "勞基法規定加班費怎麼算？",

    # 時事與閒聊類
    "馬斯克為什麼買了 Twitter？",
    "OpenAI 最新發布了什麼模型？",
    "幫我寫一篇英文自我介紹",
    "怎麼背英文單字最有效？",
    "幫我推薦一本小說",
    "世界上最高的山是哪座？",
]

# 統一拒答的格式
REFUSAL_ANSWER = "嗯⋯這題難倒我了，您的知識庫裡暫時找不到相關筆記。建議您可以: \n1. 換個關鍵字重新提問\n2. 聯絡開發人員將相關資料補充到 HackMD 後再同步知識庫"

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

    prompt = f"""以下是一段來自個人技術筆記的內容：
        {chunk_text}

        請根據這段筆記內容，生成 2 個使用者可能會問的問題，難度要有所區分：
        - 第 1 個問題：簡單問題，適合初學者，問的是基本概念或定義（例如「什麼是 X？」「X 的用途是什麼？」）
        - 第 2 個問題：困難問題，適合有一定基礎的人，問的是深層原理、比較、實作細節、或踩坑場景（例如「X 和 Y 的差別是什麼？」「在 Z 情況下 X 該怎麼處理？」）

        要求：
        - 問題要自然，像真實使用者在提問，不要過於學術
        - 問題必須能從這段筆記內容中找到答案或線索
        - 直接輸出問題，每行一個，不要編號或其他格式

        只輸出 2 個問題，不要有其他說明。
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
    prompt = f"""這是我為您查詢知識庫取得到的資訊，以下回答根據個人 HackMD 技術筆記內容整理而來，請根據以下筆記片段，針對問題給出完整且詳細的回答。
        筆記內容：
        {chunk_text}

        問題：{question}

        請嚴格遵守以下回答格式：
        1. 【結論】用 1-2 句話直接給出最核心的答案，讓人一眼看懂重點
        2. 【說明】根據筆記內容展開解釋，可包含：
        - 概念定義或背景說明
        - 具體操作步驟或程式碼範例（如果筆記有提到）
        - 生活化比喻或類比（幫助理解抽象概念）
        3. 【注意事項】補充以下任一或多項：
        - 常見錯誤或踩坑點
        - 使用限制或適用條件
        - 與其他相似概念的比較

        回答語氣要專業但不冷漠，像在幫朋友解答技術問題一樣。
        如果筆記內容不足以完整回答此問題，請直接說明「嗯⋯這題難倒我了，您的知識庫裡暫時找不到相關筆記。建議您可以: \n1. 換個關鍵字重新提問\n2. 聯絡開發人員將相關資料補充到 HackMD 後再同步知識庫」。
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